import matplotlib.pyplot as plt
import numpy as np
import plot_utils as pltu

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plot_rates(pfcmd, rates, config):
    PFCrates, MDinputs, MDrates, Outrates, Inputs, Targets, MSEs= rates
    # these tensors are  training_i x tsteps x no_neuron 
    p = config.Nsub//2
    tpb = config.trials_per_block
    Ntrain = PFCrates[:,:, :5].shape[0]
    yticks = (0, 0.5,1)
    xticks = [0, 1000, 2000]
    pfcmd.figRates, axes = plt.subplots(4,3)#, sharex=True)# , sharey=True)
    pfcmd.figRates.set_size_inches([9,7])
    ax = axes[0,0]
    ax.plot(range(Ntrain),np.mean( PFCrates[:,:,:5], axis=1), '.', markersize =0.5)
    # ax.plot(range(Ntrain), np.mean( PFCrates[:, :,:p] , axis=(1,2)), '-', linewidth=-.5)
    pltu.beautify_plot(ax,x0min=False,y0min=False, yticks=yticks, xticks=xticks)
    pltu.axes_labels(ax,'','Mean FR')
    ax.set_ylim([0,1])
    ax.set_title('PFC Up-V1')
    
    ax = axes[0,1]
    ax.plot(range(Ntrain),np.mean( PFCrates[:, :,p:p+5], axis=1), '.', markersize =0.5)
    # ax.plot(range(Ntrain), np.mean( PFCrates[:, :,p:p*2] , axis=(1,2)), '-', linewidth=-.5)
    pltu.beautify_plot(ax,x0min=False,y0min=False, yticks=yticks, xticks=xticks)
    pltu.axes_labels(ax,'','')
    ax.set_ylim([0,1])
    ax.set_title('PFC Up-V2')
    ax = axes[0,2]
    ax.plot(range(Ntrain),np.mean( PFCrates[:, :,p*2:p*2+5], axis=1), '.', markersize =0.5)
    # ax.plot(range(Ntrain), np.mean( PFCrates[:, :,p*2:p*3] , axis=(1,2)), '-', linewidth=0.5)
    pltu.beautify_plot(ax,x0min=False,y0min=False)
    ax.set_ylim([0,1])
    pltu.axes_labels(ax,'','')
    ax.set_title('PFC Down-V1')

    ninp = np.array(Inputs)
    ax = axes[1,0]
    #plot trials with up cue or down cue with blue or red.
    ax.plot(np.arange(0,Ntrain)[ninp[:,0]==1.],np.mean( MDrates[:,:,0][ninp[:,0]==1.], axis=1), '.', markersize =0.5, color='tab:blue', label='Up')
    ax.plot(np.arange(0,Ntrain)[ninp[:,0]==0.],np.mean( MDrates[:,:,0][ninp[:,0]==0.], axis=1), '.', markersize =0.5, color='tab:red',  label='Down')
    ax.legend()
    # ax.plot(range(Ntrain),np.mean( MDrates[:,:,0], axis=1), '.', markersize =0.5)
    pltu.beautify_plot(ax,x0min=False,y0min=False, yticks=yticks, xticks=xticks)
    pltu.axes_labels(ax,'','mean FR')
    ax.set_title('MD 0')
    if (len(config.variable_trials_per_block) > 0):
        for ib in range(len(config.variable_trials_per_block)-1):
            xmin = config.variable_trials_per_block[ib]
            xmax = config.variable_trials_per_block[ib+1]
            ax.axvspan(xmin, xmax, alpha=0.1, color='grey')
    else:
        for ib in range(1, config.Nblocks,2):
            ax.axvspan(tpb* ib, tpb*(ib+1), alpha=0.1, color='grey')

    ax = axes[1,1]
    ax.plot(range(Ntrain),np.mean( MDrates[:,:,1], axis=1), '.', markersize =0.5)
    pltu.beautify_plot(ax,x0min=False,y0min=False, yticks=yticks, xticks=xticks)
    pltu.axes_labels(ax,'','')
    ax.set_title('MD 1')
    
    ax = axes[1,2]
    ax.plot(range(Ntrain),np.mean( MDinputs[:, :,:], axis=1), '.', markersize =0.5)
    pltu.beautify_plot(ax,x0min=False,y0min=False, xticks=xticks)
    pltu.axes_labels(ax,'','')
    ax.set_title('MD avg inputs')
    if (len(config.variable_trials_per_block) > 0):
        for ib in range(len(config.variable_trials_per_block)-1):
            xmin = config.variable_trials_per_block[ib]
            xmax = config.variable_trials_per_block[ib+1]
            ax.axvspan(xmin, xmax, alpha=0.1, color='grey')
    else:
        for ib in range(1, config.Nblocks,2):
            ax.axvspan(tpb* ib, tpb*(ib+1), alpha=0.1, color='grey')
    
    # ax = axes[2,0]
    # ax.plot(range(Ntrain),np.mean( Outrates[:,:,0], axis=1), '.', markersize =0.5)
    # pltu.beautify_plot(ax,x0min=False,y0min=False, yticks=yticks, xticks=xticks)
    # pltu.axes_labels(ax,'','mean FR')
    # ax.set_title('Out 0')
    
    # ax = axes[2,1]
    # ax.plot(range(Ntrain),np.mean( Outrates[:,:,1], axis=1), '.', markersize =0.5)
    # pltu.beautify_plot(ax,x0min=False,y0min=False, yticks=yticks, xticks=xticks)
    # pltu.axes_labels(ax,'','')
    # ax.set_title('Out 1')
    
    # ax = axes[2,2]
    # ax.plot(range(Ntrain),np.mean( Outrates[:, :,:], axis=1), '.', markersize =0.5)
    # pltu.beautify_plot(ax,x0min=False,y0min=False, yticks=yticks, xticks=xticks)
    # pltu.axes_labels(ax,'','')
    # ax.set_title('Out 0 and 1')

    ax = axes[3,0]
    # Plot MSE
    ax.plot(MSEs)
    ax.plot(smooth(MSEs, 8), 'tab:orange', linewidth= pltu.linewidth)
    pltu.beautify_plot(ax,x0min=False,y0min=False, xticks=xticks)
    pltu.axes_labels(ax,'Trials','MSE')
    ax.set_title('MSE')

    # ax.plot(range(Ntrain),Inputs[:, 0] + np.random.uniform(-0.1, 0.1, size=(Ntrain,1)) , 'o', markersize =0.5)
    # ax.plot(range(Ntrain),Targets[:, 0] + np.random.uniform(-0.1, 0.1, size=(Ntrain,1)) , 'o', markersize =0.5)
    # pltu.beautify_plot(ax,x0min=False,y0min=False)
    # pltu.axes_labels(ax,'','Inputs')
    

    # pfcmd.figOuts = plt.figure()
    # ax = pfcmd.figOuts.add_subplot(311)
    # ax.plot(range(Ntrain), 1.*(Targets[:,0] == out_higher_mean) -0.3+ np.random.uniform(-0.01, 0.01, size=(Ntrain,) ) , '.', markersize =0.5)
    # ax.set_title('Percent correct answers smoothened over 20 trials')
    # ax = pfcmd.figOuts.add_subplot(312)
    # ax.plot(smooth((Targets[:,0] == out_higher_mean)*1., 20), linewidth=pltu.linewidth)
    # pltu.axes_labels(ax, 'Trials', '% Correct')
    # out_higher_endFR =1.*( Outrates[:, -1 ,0] >  Outrates[:, -1 ,1]                                )
    out_higher_mean = 1.*( np.mean( Outrates[:, :,0], axis=1) > np.mean( Outrates[:, :,1], axis=1) )

    Matches =  1. * (Targets[:,0] == Inputs[:,0])  # Targets is [n_trials x 2] Inputs is N_trials x 4 (cue and q values)
    Responses= 1.* (out_higher_mean == Inputs[:,0]) #* 0.8 + 0.1     #+ np.random.uniform(-noise, noise, size=(Ntrain,) )
    Corrects = 1. * (Targets[:,0] == out_higher_mean)
    Matches = Matches *1.6-0.28
    Responses = Responses *1.2-0.1

    stages = 4
    no_trials_to_score = 100

    if len(config.variable_trials_per_block) > 0:
	    tpb = 10 # NOTE: This is a hack just to get the code to run with variable
    pfcmd.score =  [np.mean(Corrects[istage*tpb:(istage*tpb)+no_trials_to_score])* 100. for istage in range(1, stages+1)]  # score binnged into stages
    pfcmd.score.append(np.mean(pfcmd.score[:-1]))   # The avrg of the cognitive flex measures, except the last forced switch block.
    pfcmd.score.append(np.mean(Corrects) * 100. )   # Add a var that holds the score of the model. % correct response. Later to be outputed as a text file.
    pfcmd.corrects = Corrects
    
    noise = 0.15
    ax = axes[3,1]
    ax.plot(Matches  + np.random.uniform(-noise, noise, size=(Ntrain,) ),  'o', markersize = 0.25, alpha=0.7)
    ax.plot(Responses+ np.random.uniform(-noise, noise, size=(Ntrain,) ),  'o', markersize = 0.25, alpha=0.7)
    pltu.axes_labels(ax, 'Trials', 'non-match    Match')
    # ax.set_title('Blue o: Correct    Orange x: response')
    ax.set_ylim([-0.3, 1.3])
    # ax.set_xlim([0, 2200])
    
    ax = axes[3,2] # Firing rates distribution
    # print('Shape is: ', PFCrates.shape)
    ax.hist(PFCrates[900:1000].flatten(), alpha=0.7 )   #, 'tab:blue') # take a slice from context 1 #[traini, tstep, Nneur] 
    ax.hist(PFCrates[2000:2100].flatten(), alpha= 0.5)  #, 'tab:red') # context 0  
    pltu.axes_labels(ax, 'rates', 'freq')


    # PLOT BEHAVIOR MEASURES
    pfcmd.figOuts = plt.figure()
    pfcmd.figOuts.set_size_inches([9,7])

    noise = 0.07
    ax = pfcmd.figOuts.add_subplot(211)
    ax.plot(Matches + np.random.uniform(-noise, noise, size=(Ntrain,)  ),    'o', markersize = 0.25, alpha=0.8)
    ax.plot(Responses+ np.random.uniform(-noise, noise, size=(Ntrain,) ),  'o', markersize = 0.25, alpha=0.8)
    pltu.axes_labels(ax, 'Trials', 'non-match     V1     Match')
    ax.set_title('Blue: Correct    Orange: response')
    ax.set_ylim([-0.8, 1.8])
    
    rm = np.convolve(Corrects, np.ones((40,))/40, mode='same')
    ax.plot(rm, color='black', linewidth= 0.5, alpha = 0.8)
    ax.plot(Inputs[:,2], color='tab:red', alpha=0.7, linewidth=0.5)
    
    for bi in range(config.Nblocks):
        plt.text((1/(config.Nblocks+1))* (0.74+bi), 0.1, str(config.block_schedule[bi]), transform=ax.transAxes)
    
    
    ax = pfcmd.figOuts.add_subplot(212)
    ax.plot(Inputs[:,4]*.1, color='tab:red', alpha=0.7,   linewidth=0.5, label='cx=match')
    st = tpb*min(4, config.Nblocks-1) - 10
    d = 30
    ax.plot(range(st, st+d), Inputs[st:st+d,5], 'o', markersize= 2, linewidth=0.5, color='tab:blue', alpha=0.7,   label='sw_dots')
    ax.plot(Inputs[:,5], color='tab:green', alpha=0.7, linewidth=0.5, label='p(sw)')
    ax.plot(Inputs[:,6], color='tab:blue', alpha=0.5, linewidth=0.5, label='p(r)')
    
    for bi in range(config.Nblocks): # LABEL contexts
        plt.text((1/13)* (0.74+bi), 0.1, str(config.block_schedule[bi]), transform=ax.transAxes)
    for ib in range(1, config.Nblocks,2): # demarcate contexts with grey shading
        ax.axvspan(tpb* ib, tpb*(ib+1), alpha=0.1, color='grey')
    
    ax.legend()

    # ax.plot(Matches,    'o', markersize = 3)
    # ax.plot(Responses,  'x', markersize = 3)
    # pltu.axes_labels(ax, 'Trials', 'non-match    Match')
    # ax.set_ylim([-0.3, 1.3])
    # ax.set_xlim([1970, 2050])

    plt.text(0.01, -0.1, str(config.args_dict), transform=ax.transAxes)
    
    # fig, axx = plt.subplots(3,1)
    # ax = axx[0]
    # t = tpb*min(3, config.Nblocks-1) - 10
    # d = 30
    # ax.plot(range(t, t+d), Inputs[t:t+d,6], 'o', markersize= 1, linewidth=0.5, color='tab:blue', alpha=0.7,   label='sm_dots')
    # ax = axx[1]
    # ax.plot(range(t, t+d), Inputs[t:t+d,6], 'o', markersize= 1, linewidth=0.5, color='tab:blue', alpha=0.7,   label='sm_dots')
    
    # ax = axx[2]
    # d = 50
    # ax.plot(range(t, t+d), Inputs[t:t+d,6], 'o', markersize= 1, linewidth=0.5, color='tab:blue', alpha=0.7,   label='sm_dots')
    
    # fig.savefig('./results/switch_signal.png')

    return # NOTE: Sabrina trying to hack

    pfcmd.figRates
    pfcmd.figRates.tight_layout()

    # PLOT within trial activity for 4 selected trials:
    trials_to_draw = [0,config.trials_per_block, config.trials_per_block+100]# [0, config.trials_per_block, int(config.Nblocks//4*config.trials_per_block)]
    pfcmd.figTrials, axes = plt.subplots(5,len(trials_to_draw))#, sharex=True)# , sharey=True)
    pfcmd.figTrials.set_size_inches([9,3*len(trials_to_draw)])
    
    for ti, trial in enumerate(range(len(trials_to_draw))):
        ax = axes[0,ti]
        ax.plot(range(200),np.mean( PFCrates[trial,:,:p], axis=1), '-', linewidth=1)
        ax.plot(range(200), PFCrates[trial,:,:5], '-', linewidth=0.5)
        pltu.beautify_plot(ax,x0min=False,y0min=False, yticks=yticks, xticks=xticks)
        pltu.axes_labels(ax,str(Inputs[trial]),'PFC Up-V1')
        ax.set_title('PFC Up-V1')
    
        ax = axes[1,ti]
        ax.plot(range(200),np.mean( PFCrates[trial,:,p:p*2], axis=1), '-', linewidth=1)
        ax.plot(range(200), PFCrates[trial,:,p:p+5], '-', linewidth=0.5)
        pltu.beautify_plot(ax,x0min=False,y0min=False, yticks=yticks, xticks=xticks)
        pltu.axes_labels(ax,str(Targets[trial]),'Up-V2')
        # ax.set_title('PFC Up-V2')

        ax = axes[2,ti]
        ax.plot(range(200),np.mean( PFCrates[trial,:,p*2:p*3], axis=1), '-', linewidth=1)
        ax.plot(range(200), PFCrates[trial,:,2*p:2*p+5], '-', linewidth=0.5)
        pltu.beautify_plot(ax,x0min=False,y0min=False, yticks=yticks, xticks=xticks)
        pltu.axes_labels(ax,'','Down-V1')
        # ax.set_title('PFC Down-V1')

        ax = axes[3,ti]
        ax.plot(range(200), MDrates[trial,:,:], '-', linewidth=1, alpha=0.7)
        ax.plot(range(200), MDinputs[trial,:,:], '-.', linewidth=2, alpha=0.7)
        pltu.beautify_plot(ax,x0min=False,y0min=False, yticks=yticks, xticks=xticks)
        pltu.axes_labels(ax,'','FR')
        ax.set_title('MD 0 and 1')

        ax = axes[4,ti]
        ax.plot(range(200), Outrates[trial,:,:], '-', linewidth=1, alpha=0.7)
        pltu.beautify_plot(ax,x0min=False,y0min=False, yticks=yticks, xticks=xticks)
        pltu.axes_labels(ax,'ms','FR')
        ax.set_title('Out 0 and 1')

    
    

def plot_weights(pfcmd, weights, config):
    wOuts, wPFC2MDs, wMD2PFCs, wMD2PFCMults, wJrecs, MDpreTraces = weights
    xticks = [0, 1000, 2000, 3000, 4000]
    # plot output weights evolution
    pfcmd.figWeights, axes = plt.subplots(5,3)#, sharex=True) #, sharey=True)
    # pfcmd.figWeights.set_figheight = pltu.twocolumnwidth
    # pfcmd.figWeights.set_figwidth = pltu.twocolumnwidth
    pfcmd.figWeights.set_size_inches([9,9])
    plot_cue_v_subpop = True
    tpb = config.trials_per_block
    if plot_cue_v_subpop:
        subplot_titles = ['Up-V1', 'Up-V2', 'Down-V1']
        p = config.Nsub//2
    else:
        subplot_titles = ['PFC cue 1', 'PFC cue 2', 'PFC cue 3']
        p = config.Nsub
    for pi, PFC in enumerate(subplot_titles):
        ax = axes[0,pi]
        ax.plot(wOuts[:,0, p*pi:p*pi+5],'tab:red', linewidth= pltu.linewidth)
        ax.plot(wOuts[:,1, p*pi:p*pi+5],'tab:blue', linewidth= pltu.linewidth)
        
        wmean = np.mean(wOuts[:,1,p*pi:p*pi+p], axis=1)
        wstd = np.mean(wOuts[:,1,p*pi:p*pi+p], axis=1)
        ax.plot(range(len(wmean)), wmean)
        ax.fill_between(range(len(wmean)), wmean-wstd, wmean+wstd, alpha=.4)

        pltu.beautify_plot(ax,x0min=False,y0min=False, xticks=xticks)
        if pi == 0: pltu.axes_labels(ax,'','to Out-0 & 1 (r,b)')
        ax.set_title(PFC)

        if (len(config.variable_trials_per_block) > 0):
            for ib in range(len(config.variable_trials_per_block)-1):
                xmin = config.variable_trials_per_block[ib]
                xmax = config.variable_trials_per_block[ib+1]
                ax.axvspan(xmin, xmax, alpha=0.1, color='grey')
        else:
            for ib in range(1, config.Nblocks,2):
                ax.axvspan(tpb* ib, tpb*(ib+1), alpha=0.1, color='grey')

    for pi, PFC in enumerate(subplot_titles):
        ax = axes[1,pi]
        ax.plot(wPFC2MDs[:,0, p*pi:p*pi+5],'tab:red', linewidth= pltu.linewidth)
        ax.plot(wPFC2MDs[:,1, p*pi:p*pi+5],'tab:blue', linewidth= pltu.linewidth)

        wmean = np.mean(wPFC2MDs[:,1,p*pi:p*pi+p], axis=1)
        wstd = np.mean(wPFC2MDs[:,1,p*pi:p*pi+p], axis=1)
        ax.plot(range(len(wmean)), wmean)
        ax.fill_between(range(len(wmean)), wmean-wstd, wmean+wstd, alpha=.4)

        pltu.beautify_plot(ax,x0min=False,y0min=False, xticks=xticks)
        if pi == 0: pltu.axes_labels(ax,'','to MD-0(r) 1(b)')

        if (len(config.variable_trials_per_block) > 0):
            for ib in range(len(config.variable_trials_per_block)-1):
                xmin = config.variable_trials_per_block[ib]
                xmax = config.variable_trials_per_block[ib+1]
                ax.axvspan(xmin, xmax, alpha=0.1, color='grey')
        else:
            for ib in range(1, config.Nblocks,2):
                ax.axvspan(tpb* ib, tpb*(ib+1), alpha=0.1, color='grey')

        ax = axes[2,pi]
        ax.plot(wMD2PFCs[:,p*pi:p*pi+5, 0],'tab:red', linewidth= pltu.linewidth)
        ax.plot(wMD2PFCs[:,p*pi:p*pi+5, 1],'tab:blue', linewidth= pltu.linewidth)
        pltu.beautify_plot(ax,x0min=False,y0min=False, xticks=xticks)
        if pi == 0: pltu.axes_labels(ax,'','from MD-0(r) 1(b)')
        if (len(config.variable_trials_per_block) > 0):
            for ib in range(len(config.variable_trials_per_block)-1):
                xmin = config.variable_trials_per_block[ib]
                xmax = config.variable_trials_per_block[ib+1]
                ax.axvspan(xmin, xmax, alpha=0.1, color='grey')
        else:
            for ib in range(1, config.Nblocks,2):
                ax.axvspan(tpb* ib, tpb*(ib+1), alpha=0.1, color='grey')

        # plot PFC to MD pre Traces
        ax = axes[3,pi]
        ax.plot(MDpreTraces[:,p*pi:p*pi+5], linewidth = pltu.linewidth)
        ax.plot(config.MDlearningBiasFactor*np.mean(MDpreTraces, axis=1), '-.', linewidth = 2)
        pltu.beautify_plot(ax,x0min=False,y0min=False, xticks=xticks)
        pltu.axes_labels(ax,'Trials','pre')
        if (len(config.variable_trials_per_block) > 0):
            for ib in range(len(config.variable_trials_per_block)-1):
                xmin = config.variable_trials_per_block[ib]
                xmax = config.variable_trials_per_block[ib+1]
                ax.axvspan(xmin, xmax, alpha=0.1, color='grey')
        else:
            for ib in range(1, config.Nblocks,2):
                ax.axvspan(tpb* ib, tpb*(ib+1), alpha=0.1, color='grey')
    
    ax = axes [4,pi]
    # ax.hist(1.+wMD2PFCMults[:,p*pi:p*pi+p, 0].flatten(), alpha=0.7 )#, 'tab:blue') # take a slice from context 1 #[traini, tstep, Nneur] 
    # ax.hist(1.+wMD2PFCMults[:,p*pi:p*pi+p, 1].flatten(), alpha=0.4 )#, 'tab:blue') # take a slice from context 1 #[traini, tstep, Nneur] 
    # pltu.axes_labels(ax, 'mul w values', 'freq')

    ax = axes [4,0] # Need ato monitor MDpretrace in a V1 vs V2 context, but also Up and Down trials. You catch the first four trials in V1 and find up and down, the the first couple in V2, get up and down. 
    ax.hist(MDpreTraces[0,p*0:p*0+p].flatten(), alpha=0.7 )#, 'tab:blue') # take a slice from context 1 #[traini, tstep, Nneur] 
    ax.hist(MDpreTraces[0,p*1:p*1+p].flatten(), alpha=0.7 )#, 'tab:blue') # take a slice from context 1 #[traini, tstep, Nneur] 
    pltu.axes_labels(ax, 'MDpre trial 0', 'freq')

    ax = axes [4,1] # Need ato monitor MDpretrace in a V1 vs V2 context, but also Up and Down trials. You catch the first four trials in V1 and find up and down, the the first couple in V2, get up and down. 
    ax.hist(MDpreTraces[1,p*0:p*0+p].flatten(), alpha=0.7 )#, 'tab:blue') # take a slice from context 1 #[traini, tstep, Nneur] 
    ax.hist(MDpreTraces[1,p*1:p*1+p].flatten(), alpha=0.7 )#, 'tab:blue') # take a slice from context 1 #[traini, tstep, Nneur] 
    pltu.axes_labels(ax, 'MDpre trial 1', 'freq')
    
    ax = axes [4,2]
    ax.hist(MDpreTraces[tpb,p*0:p*0+p].flatten(), alpha=0.7 )#, 'tab:blue') # take a slice from context 1 #[traini, tstep, Nneur] 
    ax.hist(MDpreTraces[tpb,p*1:p*1+p].flatten(), alpha=0.7 )#, 'tab:blue') # take a slice from context 1 #[traini, tstep, Nneur] 
    pltu.axes_labels(ax, 'MDpre trial tpb', 'freq')
    # ax.hist(wOuts[-1,:,:].flatten(), 50, alpha=1. )#, 'tab:blue') # take a slice from context 1 #[traini, tstep, Nneur] 
    # pltu.axes_labels(ax, 'w outs', 'freq')
    
    # axes[0,0].plot(wOuts[:,0,:5],'tab:red', linewidth= pltu.linewidth)
    # axes[0,0].plot(wOuts[:,1,:5],'tab:red', linewidth= pltu.linewidth)
    # pltu.beautify_plot(axes[0,0],x0min=False,y0min=False)
    # pltu.axes_labels(axes[0,0],'Trials','wAto0(r) wAto1(b)')
    # axes[0,1].plot(wOuts[:,0,config.Nsub:config.Nsub+5],'tab:red', linewidth= pltu.linewidth)
    # axes[0,1].plot(wOuts[:,1,config.Nsub:config.Nsub+5],'tab:red', linewidth= pltu.linewidth)
    # pltu.beautify_plot(axes[0,1],x0min=False,y0min=False)
    # pltu.axes_labels(axes[0,1],'Trials','wBto0(r) wBto1(b)')
    # axes[0,2].plot(wOuts[:,0,config.Nsub*2:config.Nsub*2+5],'tab:red', linewidth= pltu.linewidth)
    # axes[0,2].plot(wOuts[:,1,config.Nsub*2:config.Nsub*2+5],'tab:red', linewidth= pltu.linewidth)
    # pltu.beautify_plot(axes[0,2],x0min=False,y0min=False)
    # pltu.axes_labels(axes[0,2],'Trials','wCto0(r) wCto1(b)')
    # # pfcmd.figWeights.tight_layout()

    # if config.MDlearn:
    #     # plot PFC2MD weights evolution
    #     # pfcmd.figWeights = plt.figure(
    #                     # figsize=(pltu.twocolumnwidth,pltu.twocolumnwidth),
    #                     # facecolor='w')
    #     axes[1,0].plot(wPFC2MDs[:,0,:5],'tab:red', linewidth= pltu.linewidth)
    #     axes[1,0].plot(wPFC2MDs[:,0,config.Nsub*2:config.Nsub*2+5],'tab:red', linewidth= pltu.linewidth)
    #     pltu.beautify_plot(axes[1,0],x0min=False,y0min=False)
    #     pltu.axes_labels(axes[1,0],'','A -> MD0(r) C (b)')
    #     axes[1,1].plot(wPFC2MDs[:,1,:5],'tab:red', linewidth= pltu.linewidth)
    #     axes[1,1].plot(wPFC2MDs[:,1,config.Nsub*2:config.Nsub*2+5],'tab:red', linewidth= pltu.linewidth)
    #     pltu.beautify_plot(axes[1,1],x0min=False,y0min=False)
    #     pltu.axes_labels(axes[1,1],'','wA->MD1(r) C->MD1(b)')
    if config.reinforceReservoir:
        axes[1,2].plot(wJrecs[:,1,:5],'tab:red', linewidth= pltu.linewidth)
        axes[1,2].plot(wJrecs[:,-1,-5:],'tab:red', linewidth= pltu.linewidth)
        pltu.beautify_plot(axes[1,2],x0min=False,y0min=False)
        pltu.axes_labels(axes[1,2],'Trials','wRec1(r) wRec40(b)')

        # plot MD2PFC weights evolution
        # pfcmd.figWeights = plt.figure(
                        # figsize=(pltu.columnwidth,pltu.columnwidth), 
                        # facecolor='w')
        axes[2,0].plot(wMD2PFCs[:,:5,0],'r')
        axes[2,0].plot(wMD2PFCs[:,config.Nsub*2:config.Nsub*2+5,0],'tab:red', linewidth= pltu.linewidth)
        pltu.beautify_plot(axes[2,0],x0min=False,y0min=False)
        pltu.axes_labels(axes[2,0],'Trials','MD 0->A (r) 0->C (b)')
        axes[2,1].plot(wMD2PFCMults[:,:5,0],'tab:red', linewidth= pltu.linewidth)
        axes[2,1].plot(wMD2PFCMults[:,config.Nsub*2:config.Nsub*2+5,0],'tab:red', linewidth= pltu.linewidth)
        pltu.beautify_plot(axes[2,1],x0min=False,y0min=False)
        pltu.axes_labels(axes[2,1],'Trials','Mw MD0toA(r) 0->C (b)')
        # config.figWeights.tight_layout()
        axes[3,0].plot(wMD2PFCs[:,:5,0],'tab:red', linewidth= pltu.linewidth)
        axes[3,0].plot(wMD2PFCs[:,config.Nsub*2:config.Nsub*2+5,0],'tab:red', linewidth= pltu.linewidth)
        pltu.beautify_plot(axes[3,0],x0min=False,y0min=False)
        pltu.axes_labels(axes[3,0],'Trials','MD 1->A (r) 1->C (b)')
        axes[3,1].plot(wMD2PFCMults[:,:5,0],'tab:red', linewidth= pltu.linewidth)
        axes[3,1].plot(wMD2PFCMults[:,config.Nsub*2:config.Nsub*2+5,0],'tab:red', linewidth= pltu.linewidth)
        pltu.beautify_plot(axes[3,1],x0min=False,y0min=False)
        pltu.axes_labels(axes[3,1],'Trials','Mw MD1toA(r) 1->C (b)')

    pfcmd.figWeights.tight_layout()

def plot_what_i_want(pfcmd, weights, rates, config):
    PFCrates, MDinputs, MDrates, Outrates, Inputs, Targets, MSEs= rates
    wOuts, wPFC2MDs, wMD2PFCs, wMD2PFCMults, wJrecs, MDpreTraces = weights
    # these tensors are  training_i x tsteps x no_neuron 
    p = config.Nsub//2
    tpb = config.trials_per_block
    Ntrain = PFCrates[:,:, :5].shape[0]
    yticks = (0, 0.5,1)
    xticks = [0, 1000, 2000]
    pfcmd.figCustom, axes = plt.subplots(4,3)#, sharex=True)# , sharey=True)
    pfcmd.figCustom.set_size_inches([9,7])
    
    plot_trials = [0, 1, 2, 9, tpb, tpb+8, tpb+9, tpb+10, tpb*2, tpb*2+8, tpb*2+10]
    faxes = axes.flatten()
    for xi, ai in enumerate(plot_trials):
        ax = faxes[xi]
        ax.hist(MDpreTraces[ai,p*0:p*0+p].flatten(), alpha=0.7 )#, 'tab:blue') # take a slice from context 1 #[traini, tstep, Nneur] 
        ax.hist(MDpreTraces[ai,p*1:p*1+p].flatten(), alpha=0.7 )#, 'tab:blue') # take a slice from context 1 #[traini, tstep, Nneur] 
        pltu.axes_labels(ax, 'MDpre trial {}'.format(ai), 'freq')
        plt.text(0.01, 0.1, str(Inputs[ai])+ str(Targets[ai]), transform=ax.transAxes)
        ax.set_ylim([0, 15])        
        ax.set_xlim([0, .7])        
    # pfcmd.figCustom.tight_layout()




class monitor():
    # logs values for a number of model parameters, with labels, and plots them
    def __init__(self, labels):
        # Get the labels of vars to follow
        self.labels = labels
        self.vars = [[] for n in range(len(labels))]
        self.Nvars = len(labels)

    def log(self, vars):
        [self.vars[n].append(vars[n]) for n in range(len(vars))]
    def plot(self, fig, config):
        xticks = [0, 1000, 2000]
        axes = fig.subplots(4,3)#, shaqrex=True) #, sharey=True)
        fig.set_size_inches([9,7])
        p = config.Nsub
        for i, label in enumerate(self.labels):
            ax = axes.flatten()[i]
            ax.plot(self.vars[i],'tab:red', linewidth= pltu.linewidth)
            ax.set_title(label)
            pltu.beautify_plot(ax,x0min=False,y0min=False, xticks=xticks)
                        
def plot_q_values(data):
    vm_Outrates, vm_MDinputs = data
    fig, axes = plt.subplots(3,1)
    ax = axes[0]
    ax.plot(vm_Outrates.mean(axis=1))
    ax.set_title('vmPFC predictions')
    ax.legend(['v1 estimate', 'v2 est'])
    ax = axes[1]
    ax.set_title('vmPFC related MD input averages')
    ax.plot(vm_MDinputs.mean(axis=1))
    ax.legend(['MD 0 inp', 'MD 1 inp'])
    fig.savefig('./results/vmPFC.png')

def ofc_plots(error_computations, trial, name= ''):
    ## OFC plots
    figs, axes = plt.subplots(2,2)
    ax = axes[0,0]
    ax.boxplot(error_computations.wOFC2dlPFC[:,0].reshape((5, 100)).T, showmeans=True, meanline=True)
    # not entirely sure why these flips and T is necessary, but that is how it is.
    ax = axes[1,1]
    ax.boxplot(error_computations.wOFC2dlPFC[:,1].reshape((5, 100)).T, showmeans=True, meanline=True)
    ax = axes[0,1]
    ax.imshow(np.hstack([error_computations.wOFC2dlPFC[:,0].reshape((50, 10)), \
        error_computations.wOFC2dlPFC[:,1].reshape((50, 10))]))
    figs.savefig('./results/ofc_weights/OFC_w_display'+str(trial)+'_'+name+'.jpg')
    plt.close(figs)
