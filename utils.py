import numpy as np

class Logger(object):
    def __init__(self, config):
        self.config = config
        if config.save_detailed:
            self.wOuts = np.zeros(shape=(config.Ntrain, config.Nout, config.Npfc))
            self.wPFC2MDs = np.zeros(shape=(config.Ntrain, 2, config.Npfc))
            self.wMD2PFCs = np.zeros(shape=(config.Ntrain, config.Npfc, 2))
            self.wMD2PFCMults = np.zeros(shape=(config.Ntrain, config.Npfc, 2))
            self.MDpreTraces = np.zeros(shape=(config.Ntrain, config.Npfc))
            self.wJrecs = np.zeros(shape=(config.Ntrain, 40, 40)) # only save a sample 
            self.PFCrates = np.zeros((config.Ntrain, config.tsteps, config.Npfc))
            self.MDinputs = np.zeros((config.Ntrain, config.tsteps, config.Nmd))
            self.Outrates = np.zeros((config.Ntrain, config.tsteps, config.Nout))
        else:
            self.PFCrates = np.zeros((config.Ntrain,  config.Npfc))
            self.MDinputs = np.zeros((config.Ntrain,  config.Nmd))
            self.Outrates = np.zeros((config.Ntrain,  config.Nout))

        self.MDrates = np.zeros((config.Ntrain, config.tsteps, config.Nmd))
        self.Inputs = np.zeros((config.Ntrain, config.Ninputs)) 
        self.Targets = np.zeros((config.Ntrain, config.Nout))
        self.MSEs = np.zeros(config.Ntrain)

    def write_basic(self, traini, PFCrates,  MDinputs, MDrates, Outrates, Inputs, Targets, MSEs, model_obj):
        self.PFCrates[traini,  :] = PFCrates
        self.MDinputs[traini,  :] = MDinputs
        self.MDrates[traini,:, :] = MDrates
        self.Outrates[traini,  :] = Outrates
        self.Inputs[traini, :] = Inputs
        self.Targets[traini, :] = Targets
        self.MSEs[traini] = MSEs
    def write(self, traini, PFCrates,  MDinputs, MDrates, Outrates, Inputs, Targets, MSEs, model_obj):
        self.PFCrates[traini, :, :] = PFCrates
        self.MDinputs[traini, :, :] = MDinputs
        self.MDrates[traini, :, :] = MDrates
        self.Outrates[traini, :, :] = Outrates
        self.Inputs[traini, :] = Inputs
        self.Targets[traini, :] = Targets
        self.MSEs[traini] = MSEs
        if model_obj:
            if self.config.reinforceReservoir:
                # saving the whole rec is too large, 1000*1000*2200
                self.wJrecs[traini, :, :] = model_obj.Jrec[:40, 0:25:1000].detach().cpu().numpy()
            self.wOuts[traini, :, :] = model_obj.wOut
            self.wPFC2MDs[traini, :, :] = model_obj.wPFC2MD
            self.wMD2PFCs[traini, :, :] = model_obj.wMD2PFC
            self.wMD2PFCMults[traini, :, :] = model_obj.wMD2PFCMult
            self.MDpreTraces[traini, :] = model_obj.MDpreTrace


def stats(var, var_name=None):
    if type(var) == type([]): # if a list
        var = np.array(var)
    elif type(var) == type(np.array([])):
        pass #if already a numpy array, just keep going.
    else: #assume torch tensor
        pass
        # var = var.detach().cpu().numpy()
    if var_name:
        print(var_name, ':')   
    out = ('Mean, {:2.5f}, var {:2.5f}, min {:2.3f}, max {:2.3f}, norm {}'.format(var.mean(), var.var(), var.min(), var.max(),np.linalg.norm(var) ))
    print(out)
    return (out)

