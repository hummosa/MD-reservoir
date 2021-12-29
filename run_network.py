# -*- coding: utf-8 -*-
# (c) May 2018 Aditya Gilra, EPFL.

"""Extends code by Aditya Gilra. Some reservoir tweaks are inspired by Nicola and Clopath, arxiv, 2016 and Miconi 2016."""

import json
from config import Compare_to_humans_config, Config, MD_ablation_Config, OFC_control_Config, vmPFC_ablation_Config
from error_computations import Error_computations
# from refactor.ofc_trailtype import OFC as OFC_Trial
from vmPFC_k_means import OFC
from plot_figures import *
from data_generator import data_generator
import os
import numpy as np
import matplotlib as mpl

mpl.rcParams['axes.spines.left'] = True
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.bottom'] = True

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
# plt.ion()
# from IPython import embed; embed()
# import pdb; pdb.set_trace()
from scipy.io import savemat
import tqdm
import time
import plot_utils as pltu
import argparse
from model import PFCMD


def train(pfcmd, data_gen, config):
    if len(config.variable_trials_per_block) == 0:
        Ntrain = config.trials_per_block * config.Nblocks
    else: 
        Ntrain = np.sum(config.variable_trials_per_block)

    # Containers to save simulation variables
    wOuts = np.zeros(shape=(Ntrain, config.Nout, config.Npfc))
    wPFC2MDs = np.zeros(shape=(Ntrain, 2, config.Npfc))
    wMD2PFCs = np.zeros(shape=(Ntrain, config.Npfc, 2))
    wMD2PFCMults = np.zeros(shape=(Ntrain, config.Npfc, 2))
    MDpreTraces = np.zeros(shape=(Ntrain, config.Npfc))
    wJrecs = np.zeros(shape=(Ntrain, 40, 40))
    PFCrates = np.zeros((Ntrain, config.tsteps, config.Npfc))
    MDinputs = np.zeros((Ntrain, config.tsteps, config.Nmd))
    MDrates = np.zeros((Ntrain, config.tsteps, config.Nmd))
    Outrates = np.zeros((Ntrain, config.tsteps, config.Nout))
    Inputs = np.zeros((Ntrain, config.Ninputs+3)) # Adding OFC latents temp #TODO remove this.
    Targets = np.zeros((Ntrain, config.Nout))
    pfcmd.hx_of_ofc_signal_lengths = []
    MSEs = np.zeros(Ntrain)

    q_values_before = np.array([0.5, 0.5])
    for traini in tqdm.tqdm(range(Ntrain)):
        if len(config.variable_trials_per_block) > 0:
            if traini == 0:
                blocki = 0
                association_level, ofc_control = next(data_gen.block_generator(
                    blocki))  # Get the context index for this current block
            elif traini in np.cumsum(config.variable_trials_per_block):
                blocki = blocki + 1
                association_level, ofc_control = next(data_gen.block_generator(
                    blocki))  # Get the context index for this current block
        elif traini % config.trials_per_block == 0:
            blocki = traini // config.trials_per_block
            association_level, ofc_control = next(data_gen.block_generator(
                blocki))  # Get the context index for this current block
        if config.debug:
            print('context i: ', association_level)

        cue, target = data_gen.trial_generator(association_level)

        # trigger OFC switch signal for a number of trials in the block
        # q_values_before = ofc.get_v()
        error_computations.Sabrina_Q_values = ofc.get_v() # TODO: this is just a temp fix to get estimates from Sabrina's vmPFC.

        _, routs, outs, MDouts, MDinps, errors = \
            pfcmd.run_trial(association_level, q_values_before, error_computations, cue, target, config, MDeffect=config.MDeffect,
                            train=config.train)

        switch = error_computations.update_v(cue, outs, target, MDouts.mean(axis=0), routs.mean(axis=0))
        config.ofc_to_MD_gating_variable = config.ofc_effect_momentum * config.ofc_to_MD_gating_variable  # config.ofc_to_MD_gating_variable decays exponentially.
        if switch and (ofc_control == 'on'): 
            config.ofc_to_MD_gating_variable = config.ofc_effect_magnitude  #whenever a switch occurs, config.ofc_to_MD_gating_variable is reset to high value ofc_effect_magnitude

        # if traini%250==0: ofc_plots(error_computations, traini, '_')

        ofc_signal = ofc.update_v(cue, outs[-1,:], target)
        if ofc_signal == "SWITCH":
            ofc.switch_context()
        q_values_after = ofc.get_v()
        q_values_before = ofc.get_v()

        # Collect variables for analysis, plotting, and saving to disk
        area_to_plot = pfcmd
        PFCrates[traini, :, :] = routs
        MDinputs[traini, :, :] = MDinps
        MDrates[traini, :, :] = MDouts
        Outrates[traini, :, :] = outs
        Inputs[traini, :] = np.concatenate([cue, q_values_after, error_computations.p_sm_snm_ns])
        Targets[traini, :] = target
        wOuts[traini, :, :] = area_to_plot.wOut
        wPFC2MDs[traini, :, :] = area_to_plot.wPFC2MD
        wMD2PFCs[traini, :, :] = area_to_plot.wMD2PFC
        wMD2PFCMults[traini, :, :] = area_to_plot.wMD2PFCMult
        MDpreTraces[traini, :] = area_to_plot.MDpreTrace
        MSEs[traini] += np.mean(errors*errors)
        if config.reinforceReservoir:
            # saving the whole rec is too large, 1000*1000*2200
            wJrecs[traini, :, :] = area_to_plot.Jrec[:40,
                                                0:25:1000].detach().cpu().numpy()

        # Saves a data file per each trial
        # TODO possible variables to add for Mante & Sussillo condition analysis:
        #   - association level, OFC values
        if config.args_dict["save_data_by_trial"]:
            trial_weights = {
                "w_outputs": wOuts[traini].tolist(),
                "w_PFC2MD": wPFC2MDs[traini].tolist(),
                "w_MD2PFCs": wMD2PFCs[traini].tolist(),
                "w_MD2PFC_mults": wMD2PFCMults[traini].tolist(),
                "w_MD_pretraces": MDpreTraces[traini].tolist()
            }
            trial_rates = {
                "r_PFC": PFCrates[traini].tolist(),
                "MD_input": MDinputs[traini].tolist(),
                "r_MD": MDrates[traini].tolist(),
                "r_output": Outrates[traini].tolist(),
            }
            trial_data = {
                "input": Inputs[traini].tolist(),
                "target": Targets[traini].tolist(),
                "mse": MSEs[traini]
            }

            d = f"{config.args_dict['outdir']}/{config.args_dict['exp_name']}/by_trial"
            if not os.path.exists(d):
                os.makedirs(d)
            with open(f"{d}/{traini}.json", 'w') as outfile:
                json.dump({"trial_data": trial_data,
                            "network_weights": trial_weights,
                            "network_rates": trial_rates}, outfile)


    rates = [PFCrates, MDinputs, MDrates,
                Outrates, Inputs, Targets, MSEs]
    # plot_q_values([vm_Outrates, vm_MDinputs])
    # plot_weights(area_to_plot, weights, config)
    plot_rates(area_to_plot, rates, config)
    #plot_what_i_want(area_to_plot, weights, rates, config)
    # ofc_plots(error_computations, 2500, 'end_')
    #from IPython import embed; embed()
    dirname = config.args_dict['outdir'] +"/"+config.args_dict['exp_name']+"/"+config.args_dict['exp_type']+"/"
    # parm_summary = str(list(config.args_dict.values())[0])+"_"+str(
    #     list(config.args_dict.values())[1])+"_"+str(
    #     list(config.args_dict.values())[2])+"_"+str(list(config.args_dict.values())[5])
    
    parameters_to_summ = ['seed', 'var1', 'var2', 'var3']
    parm_summary = "".join([f"{config.args_dict[par]}_" for par in parameters_to_summ] )
    
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    def fn(fn_str): return os.path.join(dirname, 'fig_{}_{}_{}.{}'.format(
        fn_str, parm_summary, time.strftime("%m-%d_%H:%M"), config.figure_format))

    if config.plotFigs:  # Plotting and writing results. Needs cleaned up.
        # area_to_plot.figWeights.savefig(fn('weights'),  transparent=True,dpi=pltu.fig_dpi,
                                # facecolor='w', edgecolor='w', format=config.figure_format)
        area_to_plot.figOuts.savefig(fn('behavior'),  transparent=True,dpi=pltu.fig_dpi,
                                facecolor='w', edgecolor='w', format=config.figure_format)
        #area_to_plot.figRates.savefig(fn('rates'),    transparent=True,dpi=pltu.fig_dpi,
                                #facecolor='w', edgecolor='w', format=config.figure_format)
        if config.debug:
            area_to_plot.figTrials.savefig(fn('trials'),  transparent=True,dpi=pltu.fig_dpi,
                                    facecolor='w', edgecolor='w', format=config.figure_format)
            area_to_plot.fig_monitor = plt.figure()
            area_to_plot.monitor.plot(area_to_plot.fig_monitor, area_to_plot)
            area_to_plot.figCustom.savefig(
                fn('custom'), dpi=pltu.fig_dpi, facecolor='w', edgecolor='w', format=config.figure_format)
            area_to_plot.fig_monitor.savefig(
                fn('monitor'), dpi=pltu.fig_dpi, facecolor='w', edgecolor='w', format=config.figure_format)

    np.save(fn('saved_Corrects')[:-4]+'.npy', area_to_plot.corrects)
    # last minute add MD modulation by context to config, to test for it prior to analysing data. 
    config.md_context_modulation = np.dot(config.context_vector, MDrates.mean(1)[:,0] )/ np.sum(config.context_vector>0) # normalize by no of trials for each context. Taking MD neuron 0 or 1 should be equal.
    config.md_context_modulation = np.abs(config.md_context_modulation)


    np.save(fn('config')+'.npy', config)
    if config.saveData:  # output massive weight and rate files
        import pickle
        filehandler = open(fn('saved_rates')[:-4]+'.pickle', 'wb')
        pickle.dump(rates, filehandler)
        filehandler.close()
        filehandler = open(fn('saved_weights')[:-4]+'.pickle', 'wb')
        pickle.dump(weights, filehandler)
        filehandler.close()

        # np.save(os.path.join(dirname, 'Rates{}_{}'.format(parm_summary, time.strftime("%Y%m%d-%H%M%S"))), rates)
        # np.save(os.path.join(dirname, 'Weights{}_{}'.format(parm_summary, time.strftime("%Y%m%d-%H%M%S"))), weights)



###################################################################################
###################################################################################
###################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_argument("exp_name", default="temp",
                                nargs='?',  type=str, help="pass a str for experiment name")
    group = parser.add_argument(
        "seed", default=2, nargs='?',  type=float, help="simulation seed")
    
    group = parser.add_argument(
        "--var1", default=1, nargs='?', type=float, help="arg_1")
    group = parser.add_argument(
        "--var2", default=0.25, nargs='?', type=float, help="arg_2")
    group = parser.add_argument(
        "--var3", default=1.0, nargs='?', type=float, help="arg_3")
    group = parser.add_argument("--outdir", default="./results",
                                nargs='?',  type=str, help="pass a str for data directory")
    group = parser.add_argument("--save_data_by_trial", default=False,
                                nargs='?',  type=str, help="pass True to save data by trial")
    args = parser.parse_args()
    # OpenMind shared directory: "/om2/group/halassa/PFCMD-ali-sabrina"
     # redefine some parameters for quick experimentation and argument passing to python file.
     # Each type of Config for certain experimet will update only the relevant parameters from the args_dict
    args_dict = {'outdir':  args.outdir,  'seed': int(args.seed),
                # 'MDeffect': args.var1 , 'Gcompensation': args.var2, 'OFC_effect_magnitude': args.var3,
                'var1': args.var1 , 'var2': args.var2, 'var3': args.var3, # just for later retrievcal
                'exp_name': args.exp_name,
                'exp_type': ['Compare_to_human_data', 'MD_ablation', 'vmPFC_ablation', 'OFC_ablation'][3], #
                "save_data_by_trial": args.save_data_by_trial,
                'vmPFC_inputs': 'on',
                'MDeffect': True, 'MD_add_effect': False, 'MD_mul_effect': True,
                'ofc_target': 'MD', 'ofc_effect' : True, 'no_of_pfc_neurons_to_control': 500,
                } # 'MDlr': args.y,'switches': args.x,  'MDactive': args.z,

    if args_dict['exp_type'] == 'MD_ablation': 
        args_dict.update({'MD_mul_mean': 0 , 'MD_mul_std': 0}) # These are still unused. The weights mean and std calculations in the code are too complicated 
        config = MD_ablation_Config(args_dict)
        config.MDamplification = args.var1
    elif args_dict['exp_type'] == 'Compare_to_human_data':
        config = Compare_to_humans_config(args_dict) 
    elif args_dict['exp_type'] == 'vmPFC_ablation':
        config = vmPFC_ablation_Config(args_dict) 
    elif args_dict['exp_type'] == 'OFC_ablation': 
        config = OFC_control_Config(args_dict)
        # args.var1 = 2
        if args.var1 == 0: # OFC control is on oly if args.var1 is not 0
            config.ofc_control_schedule = ['off'] *40
        elif args.var1 == 1: # OFC control on, goes to MD
            pass
        elif args.var1 == 2: # OFC control is on goes to dlPFC
            config.ofc_to_md_active = False
            config.ofc_to_PFC_active = True
        elif args.var1 == 3: # OFC control is on goes to dlPFC, but MD mul effect is off
            config.ofc_to_md_active = False
            config.ofc_to_PFC_active = True
            config.allow_mul_effect = False 
        elif args.var1 == 3: # OFC control is on goes to dlPFC, but both MD mul and add effect is off
            config.ofc_to_md_active = False
            config.ofc_to_PFC_active = True
            config.allow_mul_effect = False 
            config.allow_add_effect = False 
        config.ofc_effect_magnitude = 1. 
        config.OFC2dlPFC_factor = 0.1 # OFC2dlPFC weights (with a norm of 1) need multiplied by 10 to be effective.  
        config.ofc_timesteps_active = int(args.var2) # 1 #apparantly 1 is enough. 
        config.allow_ofc_control_to_no_pfc =  config.Npfc #int(args.var2)
        config.OFC2dlPFC_lr  = 1e-3
    else: 
        config = Config(args_dict)

    ofc = OFC() # Sabrina's vmPFC model. 
    error_computations = Error_computations(config) # Baseline error computation, OFC Bayesian model,  and overall error for node perturbation learning

    # config.no_of_trials_with_ofc_signal = int(args_dict['switches'])
    # config.MDamplification = 30.  # args_dict['switches']
    # config.MDlearningBiasFactor = args_dict['MDactive']

    pfcmd = PFCMD(config)
        
    if config.reLoadWeights:
        filename = 'dataPFCMD/data_reservoir_PFC_MD' + '_R'+str(pfcmd.RNGSEED) + '.shelve'
        pfcmd.load(filename)
    t = time.perf_counter()

    train(pfcmd , data_generator(config), config)
    
    print('training_time', (time.perf_counter() - t)/60, ' minutes')
    
    if config.saveData:
        pfcmd.save()
        pfcmd.fileDict.close()

