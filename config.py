import numpy as np

class Config():
    def __init__(self, args_dict={}):
        # Current args required:
        # 'seed' # Seed for the np random number generator
        self.args_dict = args_dict
        #enviroment parameters:
        self.plotFigs = False
        self.debug = False
        self.saveData = False
        self.save_detailed = True
        # self.figure_format =  'PDF' # self.figure_format =  'EPS'
        self.figure_format =  'PNG'
        self.outdir = args_dict['outdir'] if 'outdir' in args_dict else './results/'
        self.RNGSEED = args_dict['seed'] if 'seed' in args_dict else 1

        np.random.seed([self.RNGSEED])
        self.cuda = False
        # self.args = args_dict               # dict of args label:value

        #Experiment parameters:
        self.Ntasks = 2                     # Ambiguous variable name, replacing with appropriate ones below:  # number of contexts
        self.Ncontexts = 2                  # number of contexts (match block or non-match block)
        self.trials_per_block = None #500
        self.variable_trials_per_block = [500, 500, 800, 600, 800, 600, 800, 600, 600, 800, 800, 600]
        #self.variable_trials_per_block = [500, 500, 400, 300, 400, 300, 400, 300, 300, 400, 400, 300]
        self.Nblocks = 10  # Deprecated                 # number of blocks for the simulation
       	self.block_schedule = ['10', '90'] * 6 #['30', '90', '10', '90', '70', '30', '10', '70']
        # self.block_schedule = ['90', '10', '90', '30', '50', '70', '10', '50', '90', '30', '70', '10']
        self.ofc_control_schedule = ['off'] * 14  # ['on'] *40  + ['match', 'non-match'] *1 + ['on'] *40

        # general params
        self.tau = 0.02
        self.dt = 0.001
        self.tsteps = 200                   # number of timesteps in a trial
        self.cuesteps = 100                 # number of time steps for which cue is on
        self.response_delay = 0             # time between cue end and begin response, if 0 all trial is averaged for response
        self.noiseSD = 1e-3
        self.learning_rate = 5e-6  # too high a learning rate makes the output weights change too much within a trial / training cycle,

        #Network architecture
        self.wV_structured = True
        self.Ninputs = 4                 # Total number of inputs
        self.Ncues = 2                   # How many of the inputs are task cues (UP, DOWN)
        self.Nmd    = 2                  # number of MD cells.
        self.Npfc = 500                  # number of pfc neurons
        self.Nofc = 500                  # number of ofc neurons
        self.Nsub = 200                  # number of neurons per cue
        self.Nout = 2                    # number of outputs
        self.G = 1                       # Controls level of excitation in the net
        self.reLoadWeights = False

        self.train = True   # swich training on or off.
        self.tauError = 0.001            # smooth the error a bit, so that weights don't fluctuate
        self.modular  = False                # Assumes PFC modules and pass input to only one module per tempral context.
        self.MDeffect = True                # whether to have MD present or not
        self.MDremovalCompensationFactor = 1.3 # If MD effect is removed, excitation drops, multiply recurrent connection conductance by this factor to compensate
        self.MDamplification = 30.           # Factor by which MD amplifies PFC recurrent connections multiplicatively
        self.MDamplification_add = 1.           # Ampflication factor for the add gates. 1 by default. 
        self.MDlearningrate = 5e-5 #1e-4 # 1e-7   #Separate learning rate for Hebbian plasticity at MD-PFC synapses.
        self.MDrange = 0.1                  # Allowable range for MD-PFC synapses.
        self.MDlearningBias = 0.3           # threshold for Hebbian learning. Biases pre*post activity.
        self.MDlearningBiasFactor = 1.     # Switched dynamic Bias calc based on average, this gets multiplied with running avg resulting in effective bias for hebbian learning.
        self.cueFactor = 0.5 #args_dict['CueFactor']#0.5# 0.75  1.5 Ali halved it when I added cues going to both PFC regions, i.e two copies of input. But now working ok even with only one copy of input.
        self.delayed_response = 0 #50       # in ms, Reward model based on last 50ms of trial, if 0 take mean error of entire trial. Impose a delay between cue and stimulus.

        # Model ablations:
        self.instruct_md_behavior = False   # to disable the effects of context discovery by hebbian learning. Istruct ideal MD behavior, and examine other parts of the model.
        self.allow_add_effect = True       # Set to False to ablate MD additive effects
        self.allow_mul_effect = True     # Set to False to ablate MD multiplicative effects
        self.allow_value_inputs = True     # set to false to Zero out the weights of the value inputs
        self.allow_ofc_control_to_no_pfc = self.Npfc    # Limit ofc effect to certain no of PFC cells.

        # OFC
        self.follow = 'behavioral_context' # 'association_levels'  # in estimating baseline_err whether to track each context (match, non-match) or more granularily track assocation levels
        self.horizon = 10               # how many trials to look back when calculating Q values for actions available.
        self.OFC_reward_hx = True           # model ofc as keeping track of current strategy and recent reward hx for each startegy.
        self.use_context_belief_to_switch_MD = True  # input routing per current context or per context belief
        self.no_of_trials_with_ofc_signal = 20 # (Deprecated) no of trials with OFC sparse switch control signal.
        self.ofc_to_MD_gating_variable = 0 #Gating OFC to MD or to PFC connections. Initialized to 0 (closed)
        self.ofc_to_md_active = True  # two Variables to decide whether OFC switch signal goes to MD or to PFC. (Might want to test both active at some point)
        self.ofc_to_PFC_active = False
        self.ofc_timesteps_active = 200    # num of timesteps at the begning of trial that input from OFC is applied
        self.ofc_effect_magnitude = 1.0    # THe peak of the gating value from OFC to MD. If 0, no flow of info from OFC to MD.
        self.ofc_effect_momentum = 0.9
        self.positiveRates = True           # whether to clip rates to be only positive, G must also change
        self.OFC2dlPFC_lr  = 1e-3        # learning rate for the ofc direct to dlPFC synapses.
        self.OFC2dlPFC_factor  = 0.5        # some scaling factor I had multiplied with the norm from before. Keeps norm at 2.
        self.reinforce = True              # use reinforcement learning (node perturbation) a la Miconi 2017
        if self.reinforce:                 # instead of error-driven learning
            self.learning_rate *= 10       # increase learning rate for reinforce
        self.MDreinforce = False

        self.perturbProb = 50./self.tsteps
                                        # probability of perturbation of each output neuron per time step
        self.perturbAmpl = 10.          # how much to perturb the output by
        self.meanErrors = np.zeros(self.Ncontexts)#*self.inpsPerContext) #Ali made errors per context rather than per context*cue
                                        # vector holding running mean error for each cue
        self.decayErrorPerTrial = 0.1   # how to decay the mean errorEnd by, per trial
        self.reinforceReservoir = False # learning on reservoir weights also?
        if self.reinforceReservoir:
            self.perturbProb /= 10

        self.md_context_modulation = np.nan

    @property
    def variable_trials_per_block(self):
        return(self._variable_trials_per_block)
    @variable_trials_per_block.setter
    def variable_trials_per_block(self, variable_trials_per_block):
        self._variable_trials_per_block = variable_trials_per_block
        ## create a context vector to compare signals to
        self.context_vector = np.ones(np.sum(self.variable_trials_per_block))
        vt = np.cumsum([0]+ self.variable_trials_per_block)
        if len(self.variable_trials_per_block) > 0:
            for bi in range(0, len(self.variable_trials_per_block), 2):
                self.context_vector[vt[bi]:vt[bi]+self.variable_trials_per_block[bi]] = np.ones(self.variable_trials_per_block[bi]) * -1 # use this to store the final calculated value, will be saved with the config object, avoids having to save md firing rates.
        if len(self.variable_trials_per_block) == 0:
            self.Ntrain = self.trials_per_block * self.Nblocks
        else:
            self.Ntrain = np.sum(self.variable_trials_per_block)

## Configs for common experiments.
class Compare_to_humans_config(Config):
    def __init__(self, args_dict={}):
        super().__init__(args_dict)
        self.trials_per_block = None #500
        self.variable_trials_per_block = [500, 500, 800, 600, 800, 600, 800, 600, 600, 800, 800, 600]
        #self.variable_trials_per_block = [500, 500, 400, 300, 400, 300, 400, 300, 300, 400, 400, 300]
        self.block_schedule = ['90', '10', '90', '30', '50', '70', '10', '50', '90', '30', '70', '10']
        self.ofc_control_schedule = ['off'] * 14  # ['on'] *40  + ['match', 'non-match'] *1 + ['on'] *40

class Paramater_search_Config(Config):
    def __init__(self, args_dict={'MDeffect': 'on'}):
        super().__init__(args_dict)
        if args_dict['MD_multi'] =='on':
            self.MDeffect = True
            self.MDremovalCompensationFactor = 1.0
        else:
            self.MDeffect = False
            self.MDremovalCompensationFactor = 1.3
        self.MDamplification = 30.

        self.variable_trials_per_block = [500] * 6
       	self.block_schedule = ['10', '90'] * 6 #['30', '90', '10', '90', '70', '30', '10', '70']
        self.ofc_control_schedule = ['off'] * 14  # ['on'] *40  + ['match', 'non-match'] *1 + ['on'] *40

class MD_ablation_Config(Config):
    def __init__(self, args_dict={'MDeffect': True, 'MD_add_effect': True, 'MD_mul_effect': True}):
        super().__init__(args_dict)
        if args_dict['MDeffect']:
            self.MDeffect = True
            # self.MDremovalCompensationFactor = 1.0
        else:
            self.MDeffect = False
            self.MDremovalCompensationFactor = 1.3
        self.allow_add_effect = args_dict['MD_add_effect']
        self.allow_mul_effect = args_dict['MD_mul_effect']
        # self.MD_mul_mean, self.MD_mul_std = args_dict['MD_mul_mean'], args_dict['MD_mul_std']
        # self.MD_add_mean, self.MD_add_std = args_dict['MD_add_mean'], args_dict['MD_add_std']
        self.variable_trials_per_block = [500] * 6
        self.block_schedule = ['10', '90'] * 6 #['30', '90', '10', '90', '70', '30', '10', '70']
        self.ofc_control_schedule = ['off'] * 14  # ['on'] *40  + ['match', 'non-match'] *1 + ['on'] *40

class HebbianLearning_config(Config):
    def __init__(self, args_dict={'MDrange': True, 'MDlearningrate': True, 'PreSynaptic_pretrace_time_constant': True}):
        super().__init__(args_dict)
        self.variable_trials_per_block = [500] * 6
        self.block_schedule = ['10', '90'] * 6 #* 6 #['30', '90', '10', '90', '70', '30', '10', '70']
        self.ofc_control_schedule = ['off'] * 14  # ['on'] *40  + ['match', 'non-match'] *1 + ['on'] *40

class vmPFC_ablation_Config(Config):
    def __init__(self, args_dict={}):
        super().__init__(args_dict)
        self.variable_trials_per_block = [500] * 6
        self.block_schedule = ['10', '90'] * 6 #['30', '90', '10', '90', '70', '30', '10', '70']
        self.ofc_control_schedule = ['off'] * 14  # ['on'] *40  + ['match', 'non-match'] *1 + ['on'] *40

class OFC_control_Config(Config):
    def __init__(self, args_dict={'ofc_target': 'MD', 'ofc_effect' : True, 'no_of_pfc_neurons_to_control': 500}):
        super().__init__(args_dict)
        self.ofc_control_schedule = ['of'] *4  + ['on'] * 40
        self.variable_trials_per_block = [500] * 8 + [100] * 6
       	self.block_schedule = ['10', '90'] * 12 #['30', '90', '10', '90', '70', '30', '10', '70']
        self.allow_ofc_control_to_no_pfc = self.Npfc    # Limit ofc effect to certain no of PFC cells.
        self.ofc_to_md_active = True  # two Variables to decide whether OFC switch signal goes to MD or to PFC. (Might want to test both active at some point)
        self.ofc_to_PFC_active = False
