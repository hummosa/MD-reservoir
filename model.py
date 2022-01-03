# -*- coding: utf-8 -*-
# (c) May 2018 Aditya Gilra, EPFL.

"""Extends code by Aditya Gilra. Some reservoir tweaks are inspired by Nicola and Clopath, arxiv, 2016 and Miconi 2016."""

import torch
import json
from config import Config
from error_computations import Error_computations
# from refactor.ofc_trailtype import OFC as OFC_Trial
from vmPFC_k_means import OFC
from plot_figures import *
from data_generator import data_generator
import os
import numpy as np
import matplotlib.pyplot as plt
# plt.ion()
# from IPython import embed; embed()
# import pdb; pdb.set_trace()
from scipy.io import savemat
import sys
import shelve
import tqdm
import time
import plot_utils as pltu
import argparse
cuda = False
if cuda:
    import torch


# data_generator = data_generator()



class PFCMD():
    def __init__(self, config, args_dict={}):
        # * Adjust network excitation levels based on MD effect, Positive Rates, and activation fxn
        if not config.MDeffect:
            config.G *= config.MDremovalCompensationFactor

        # I don't want to have an if inside activation  as it is called at each time step of the simulation
        # But just defining within __init__ doesn't make it a member method of the class,
        #  hence the special config.__class__. assignment
        if config.positiveRates:
            # only +ve rates
            def activation(self, inp):
                return np.clip(np.tanh(inp), 0, None)
        else:
            # both +ve/-ve rates as in Miconi
            def activation(self, inp):
                return np.tanh(inp)
        self.__class__.activation = activation
        # Choose G based on the type of activation function
        # unclipped activation requires lower G than clipped activation,
        #  which in turn requires lower G than shifted tanh activation.
        if config.positiveRates:
            config.tauMD = config.tau
        else:
            config.G /= 2.
            config.MDthreshold = 0.4
            config.tauMD = config.tau*10

        if config.saveData:
            self.fileDict = shelve.open('dataPFCMD/data_reservoir_PFC_MD' +
                                        str(config.RNGSEED) +
                                        ('')+'.shelve')

        # *# init weights:
        # MD PFC weights
        self.wPFC2MD = np.zeros(shape=(config.Nmd, config.Npfc))

        self.wPFC2MD = np.random.normal(size=(config.Nmd, config.Npfc))\
            * config.MDrange  # *config.G/np.sqrt(config.Nsub*2)
        # same as res rec, substract mean from each row.
        self.wPFC2MD -= np.mean(self.wPFC2MD, axis=1)[:, np.newaxis]
        self.wMD2PFC = np.random.normal(size=(config.Npfc, config.Nmd))\
            * config.MDrange  # *config.G/np.sqrt(config.Nsub*2)
        # same as res rec, substract mean from each row.
        self.wMD2PFC -= np.mean(self.wMD2PFC, axis=1)[:, np.newaxis]
        self.wMD2PFCMult = self.wMD2PFC  # Get the exact copy to init mult weights
        self.initial_norm_wPFC2MD = np.linalg.norm(self.wPFC2MD) * .6
        self.initial_norm_wMD2PFC = np.linalg.norm(self.wMD2PFC) * .6
        # Recurrent weights
        self.Jrec = np.random.normal(size=(config.Npfc, config.Npfc))\
            * config.G/np.sqrt(config.Nsub)
        if cuda:
            self.Jrec = torch.Tensor(self.Jrec).cuda()
        # Output weights
        self.wOut = np.random.uniform(-1, 1,
                                      size=(config.Nout, config.Npfc))/config.Npfc
        # Input weights
        self.wV = np.zeros((config.Npfc, 2))
        self.wIn = np.zeros((config.Npfc, config.Ncues))

        if config.positiveRates:
            lowcue, highcue = 0.5, 1.
        else:
            lowcue, highcue = -1., 1
        for cuei in np.arange(config.Ncues):
            self.wIn[config.Nsub*cuei:config.Nsub*(cuei+1), cuei] = \
                np.random.uniform(lowcue, highcue, size=config.Nsub) \
                * config.cueFactor * 0.8  # to match that the max diff between v1 v2 is 0.8
            if config.wV_structured:
                self.wV[config.Nsub*cuei:config.Nsub*(cuei)+config.Nsub//2, 0] = \
                    np.random.uniform(lowcue, highcue, size=config.Nsub//2) \
                    * config.cueFactor
                self.wV[config.Nsub*(cuei)+config.Nsub//2:config.Nsub*(cuei+1), 1] = \
                    np.random.uniform(lowcue, highcue, size=config.Nsub//2) \
                    * config.cueFactor

            else:
                input_variance = 1.5
                self.wV = np.random.normal(size=(config.Npfc, 2), loc=(
                    lowcue+highcue)/2, scale=input_variance) * config.cueFactor  # weights of value input to pfc
                self.wV = np.clip(self.wV, 0, 1)
                self.wIn = np.random.normal(size=(config.Npfc, config.Ncues), loc=(
                    lowcue+highcue)/2, scale=input_variance) * config.cueFactor
                self.wIn = np.clip(self.wIn, 0, 1)

        self.MDpreTrace = np.zeros(shape=(config.Npfc))

        # make mean input to each row zero, helps to avoid saturation (both sides) for positive-only rates.
        #  see Nicola & Clopath 2016 mean of rows i.e. across columns (axis 1), then expand with np.newaxis
        #   so that numpy's broadcast works on rows not columns
        if cuda:
            with torch.no_grad():
                self.Jrec -= torch.mean(self.Jrec, dim=1, keepdim=True)
        else:
            self.Jrec -= np.mean(self.Jrec, axis=1)[:, np.newaxis]

    def run_trial(self, association_level, Q_values, error_computations, cue, target, config, MDeffect=True,
                  MDCueOff=False, MDDelayOff=False,
                  train=True, routsTarget=None):
        '''
        config.reinforce trains output weights
         using REINFORCE / node perturbation a la Miconi 2017.'''
        cues = np.zeros(shape=(config.tsteps, config.Ncues))

        xinp = np.random.uniform(0, 0.1, size=(config.Npfc))
        xadd = np.zeros(shape=(config.Npfc))
        MDinp = np.random.uniform(0, 0.1, size=config.Nmd)
        MDinps = np.zeros(shape=(config.tsteps, config.Nmd))
        routs = np.zeros(shape=(config.tsteps, config.Npfc))
        MDouts = np.zeros(shape=(config.tsteps, config.Nmd))
        outInp = np.zeros(shape=config.Nout)
        outs = np.zeros(shape=(config.tsteps, config.Nout))
        out = np.zeros(config.Nout)
        errors = np.zeros(shape=(config.tsteps, config.Nout))
        error_smooth = np.zeros(shape=config.Nout)

        # init a Hebbian Trace for node perturbation to keep track of eligibilty trace.
        if config.reinforce:
            HebbTrace = np.zeros(shape=(config.Nout, config.Npfc))
            if config.reinforceReservoir:
                if cuda:
                    HebbTraceRec = torch.Tensor(
                        np.zeros(shape=(config.Npfc, config.Npfc))).cuda()
                else:

                    HebbTraceRec = np.zeros(shape=(config.Npfc, config.Npfc))
            if config.MDreinforce:
                HebbTraceMD = np.zeros(shape=(config.Nmd, config.Npfc))

        for i in range(config.tsteps):
            rout = self.activation(xinp)
            routs[i, :] = rout
            outAdd = np.dot(self.wOut, rout)

            # Gather MD inputs
            if config.ofc_to_md_active and (i < config.ofc_timesteps_active):
                input_from_ofc = np.dot(error_computations.wOFC2MD , error_computations.vec_current_context )
                MDinp += config.ofc_to_MD_gating_variable * input_from_ofc

            if config.positiveRates:
                MDinp += config.dt/config.tau * (-MDinp + np.dot(self.wPFC2MD, rout))
            else:  # shift PFC rates, so that mean is non-zero to turn MD on
                MDinp += config.dt/config.tau * 10. * (-MDinp + np.dot(self.wPFC2MD, (rout+1./2)))

            # winner take all on the MD hardcoded for config.Nmd = 2
            if MDinp[0] > MDinp[1]:
                MDout = np.array([1, 0])
            else:
                MDout = np.array([0, 1])

            ########### controlled MD behavior.
            if config.instruct_md_behavior:
                MDout = np.array([1, 0]) if association_level in ['90', '70', '50'] else np.array([0, 1])

            MDouts[i, :] = MDout
            MDinps[i, :] = MDinp

            # Gather PFC inputs

            if MDeffect:
                # Add multplicative amplification of recurrent inputs.
                if config.allow_mul_effect:
                    self.MD2PFCMult = np.dot(self.wMD2PFCMult * config.MDamplification, MDout)
                else: # to ablate multi effect, fix MD pattern
                    self.MD2PFCMult = np.dot(self.wMD2PFCMult * config.MDamplification, np.array([0, 1]))
                xadd = (1.+self.MD2PFCMult) * np.dot(self.Jrec, rout)

                # Additive MD input to PFC
                if config.allow_add_effect:
                    xadd += np.dot(self.wMD2PFC , MDout)
                else: # to ablate add effect, fix MD pattern
                    xadd += np.dot(self.wMD2PFC , np.array([0, 1]))
            else:
                xadd = np.dot(self.Jrec, rout)

            if config.ofc_to_PFC_active and (i < config.ofc_timesteps_active):
                input_from_ofc = np.dot(error_computations.wOFC2dlPFC , error_computations.vec_current_context )
                ofc_to_pfc_mask = np.zeros_like(input_from_ofc)
                ofc_to_pfc_mask[:config.allow_ofc_control_to_no_pfc] = np.ones_like(input_from_ofc)[:config.allow_ofc_control_to_no_pfc]

                xadd += config.ofc_to_MD_gating_variable * input_from_ofc * ofc_to_pfc_mask

            if i < config.cuesteps:
                # if MDeffect and useMult:
                #    xadd += self.MD2PFCMult * np.dot(self.wIn,cue)
                xadd += np.dot(self.wIn, cue)
                if config.allow_value_inputs:
                    xadd += np.dot(self.wV, Q_values)
                else:
                    xadd += np.dot(self.wV, np.array([0.5,0.5]))

            # MD Hebbian learning
            if train and not config.MDreinforce:
                # MD presynaptic traces evolve dyanamically during trial and across trials
                # to decrease fluctuations.
                self.MDpreTrace += 1./(10.*config.tsteps) * \
                    (-self.MDpreTrace + rout)
                MDlearningBias = config.MDlearningBiasFactor * \
                    np.mean(self.MDpreTrace)
                MDrange = config.MDrange  # 0.05#0.1#0.06
                # Ali changed from 1e-4 and thresh from 0.13
                wPFC2MDdelta = np.outer(
                    MDout-0.5, self.MDpreTrace-MDlearningBias)
                # Ali lowered to 0.01 from 1.
                self.wPFC2MD = np.clip(
                    self.wPFC2MD + config.MDlearningrate*wPFC2MDdelta, -MDrange, MDrange)
                # self.wMD2PFC = np.clip(self.wMD2PFC +fast_delta.T,-MDrange , MDrange ) # lowered from 10.
                # self.wMD2PFCMult = np.clip(self.wMD2PFCMult+ slow_delta.T,-2*MDrange /self.G, 2*MDrange /self.G)
                # self.wMD2PFCMult = np.clip(self.wMD2PFC,-2*MDrange /self.G, 2*MDrange /self.G) * self.MDamplification

            # Add random perturbations to neurons
            if config.reinforce:
                # Exploratory perturbations a la Miconi 2017 Perturb each output neuron independently
                #  with probability perturbProb
                perturbationOff = np.where(
                    np.random.uniform(size=config.Nout) >= config.perturbProb)
                perturbation = np.random.uniform(-1, 1, size=config.Nout)
                perturbation[perturbationOff] = 0.
                perturbation *= config.perturbAmpl
                outAdd += perturbation

                if config.reinforceReservoir:
                    perturbationOff = np.where(
                        np.random.uniform(size=config.Npfc) >= config.perturbProb)
                    perturbationRec = np.random.uniform(-1,
                                                        1, size=config.Npfc)
                    perturbationRec[perturbationOff] = 0.
                    perturbationRec *= config.perturbAmpl
                    xadd += perturbationRec

                if config.MDreinforce:
                    perturbationOff = np.where(
                        np.random.uniform(size=config.Nmd) >= config.perturbProb)
                    perturbationMD = np.random.uniform(-1, 1, size=config.Nmd)
                    perturbationMD[perturbationOff] = 0.
                    perturbationMD *= config.perturbAmpl
                    MDinp += perturbationMD

            # Evolve inputs dynamically to cells before applying activations
            xinp += config.dt/config.tau * (-xinp + xadd)
            # Add noise
            xinp += np.random.normal(size=(config.Npfc))*config.noiseSD \
                * np.sqrt(config.dt)/config.tau
            # Activation of PFC cells happens at the begnining of next timestep
            outInp += config.dt/config.tau * (-outInp + outAdd)
            out = self.activation(outInp)

            error = out - target
            errors[i, :] = error
            outs[i, :] = out
            error_smooth += config.dt/config.tauError * (-error_smooth + error)

            if train:  # Get the pre*post activity for the preturbation trace
                if config.reinforce:
                    # note: rout is the activity vector for previous time step
                    HebbTrace += np.outer(perturbation, rout)
                    if config.reinforceReservoir:
                        if cuda:
                            with torch.no_grad():
                                HebbTraceRec += torch.ger(torch.Tensor(
                                    perturbationRec).cuda(), torch.Tensor(rout).cuda())
                        else:
                            HebbTraceRec += np.outer(perturbationRec, rout)
                    if config.MDreinforce:
                        HebbTraceMD += np.outer(perturbationMD, rout)
                else:
                    # error-driven i.e. error*pre (perceptron like) learning
                    self.wOut += -config.learning_rate \
                        * np.outer(error_smooth, rout)
        # * At trial end:
        #################
        # get inferred context id from ofc
        cid = error_computations.get_cid(association_level)
        trial_err, all_contexts_err = error_computations.get_trial_err(
            errors, association_level)
        baseline_err = error_computations.baseline_err

        if train:  # and config.reinforce:
            # with learning using REINFORCE / node perturbation (Miconi 2017),
            #  the weights are only changed once, at the end of the trial
            # apart from eta * (err-baseline_err) * hebbianTrace,
            #  the extra factor baseline_err helps to stabilize learning
            #   as per Miconi 2017's code,
            #  but I found that it destabilized learning, so not using it.
            self.wOut -= config.learning_rate * \
                (trial_err-baseline_err[cid]) * \
                HebbTrace  # * baseline_err[cid]

            if config.reinforceReservoir:
                if cuda:
                    with torch.no_grad():
                        self.Jrec -= config.learning_rate * \
                            (trial_err-baseline_err[cid]) * \
                            HebbTraceRec  # * baseline_err[cid]
                else:
                    self.Jrec -= config.learning_rate * \
                        (trial_err-baseline_err[cid]) * \
                        HebbTraceRec  # * baseline_err[cid]
            if config.MDreinforce:
                self.wPFC2MD -= config.learning_rate * \
                    (trial_err-baseline_err[cid]) * \
                    HebbTraceMD * \
                    10.  # changes too small Ali amplified #* baseline_err[cid]
                self.wMD2PFC -= config.learning_rate * \
                    (trial_err-baseline_err[cid]) * \
                    HebbTraceMD.T * 10.  # * baseline_err[cid]


            # synaptic scaling and competition both ways at MD-PFC synapses.
            self.wPFC2MD /= np.linalg.norm(self.wPFC2MD) / self.initial_norm_wPFC2MD
            self.wMD2PFC /= np.linalg.norm(self.wMD2PFC) / self.initial_norm_wMD2PFC
            # stats(self.wMD2PFC, 'add')
            # stats(self.wMD2PFCMult, 'multi')

        baseline_err = error_computations.update_baseline_err(all_contexts_err)
        # self.monitor.log({'qvalue0':error_computations.Q_values[0], 'qvalue1':error_computations.Q_values[1]})

        return cues, routs, outs, MDouts, MDinps, errors


    def load(self, filename):
        d = shelve.open(filename)  # open
        self.wOut = d['wOut']
        self.wMD2PFC = d['MD2PFC']
        self.wMD2PFCMult = d['MD2PFCMult']
        self.wPFC2MD = d['PFC2MD']

        d.close()
        return None

    def save(self):
        self.fileDict['wOut'] = self.wOut
        self.fileDict['MD2PFC'] = self.wMD2PFC
        self.fileDict['MD2PFCMult'] = self.wMD2PFCMult
        self.fileDict['PFC2MD'] = self.wPFC2MD


