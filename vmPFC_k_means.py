# Sabrina's model. Uses a set of k means (k = ASSOCIATION_RANGE_N) to parametrize binomial distributions,
#  and associates them with data points

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom


class OFC:
    ASSOCIATION_RANGE_N = 41
    ASSOCIATION_RANGE = np.linspace(0, 1, ASSOCIATION_RANGE_N)
    # NOTE: STD_THRESH should be adjusted based on association level
    # For association levels 0.9 and 0.1 we use a threshold value of 2
    # because we expect tight distributions
    STD_THRESH = 1.5
    STD_WINDOW_SZ = 5
    # NOTE: The error alpha chosen is 2 because the expected error is 0.1 with
    # association levels 0.9 and 0.1 when one is using the maximizing strategy.
    # So with the window size of 5, expected error becomes 0.5
    # If more than 1 errors occurs in the window, we switch
    # NOTE: Sabrina -- Alpha may need to be changed when I integrate into
    # the complete model

    def __init__(self, switch_thresh=0.2, horizon_sz=20):
        self.switch_thresh = switch_thresh
        self.horizon_sz = horizon_sz

        n = self.ASSOCIATION_RANGE_N
        self.prior = np.ones(n) / n  # Assume a uniform prior
        self.contexts = {}

        self.has_dist_convgered = False
        self.std_history = []

        self.trial_type_history = []

    def get_v(self):
        if (len(self.prior) == 0):
            return np.array([0.5, 0.5])
        else:
            idx_MAPs = np.where(self.prior == max(self.prior))[0]
            idx_MAP = np.random.choice(idx_MAPs)
            v1 = self.ASSOCIATION_RANGE[idx_MAP]
            v2 = 1 - v1
            return np.array([v1, v2])

    def switch_context(self):
        [v1, v2] = self.get_v()
        old_ctx = np.round(v1, 1)
        self.contexts[str(old_ctx)] = self.prior

        tt = [1 if x == "MATCH" else 0 for x in self.trial_type_history]
        new_ctx = np.round(np.mean(tt), 1)
        # print(tt, new_ctx)

        # print("Switching from old ctx to new ctx", old_ctx, new_ctx)
        # print(new_ctx, str(new_ctx), self.contexts.keys())

        if str(new_ctx) in self.contexts:
            self.prior = self.contexts[str(new_ctx)]
            self.has_dist_convgered = True
            self.std_history = []
            # print("Pulling new ctx from cache...")
        else:
            # NOTE: Prior 1 -- binominal
            # n = len(self.ASSOCIATION_RANGE)
            # p = ctx
            # self.prior = np.array([binom.pmf(k, n, p) for k in range(n)])

            # NOTE: Prior 2 -- uniform
            n = len(self.ASSOCIATION_RANGE)
            self.prior = np.ones(n) / n  # Assume a uniform prior
            for trial_type in self.trial_type_history:
                self.prior = self.compute_posterior(trial_type)
            self.has_dist_convgered = False
            self.std_history = []
            # print("Initializnig a new ctx...")

    def compute_std(self):
        N = self.ASSOCIATION_RANGE_N * 1.
        mean = sum([(x/N) * p for x, p in enumerate(self.prior)])
        std = sum([p * ((x/N) - mean)**2
                   for x, p in enumerate(self.prior)]) ** 0.5

        # print('mean std', mean, std)
        return std

    def compute_trial_err(self, stimulus, choice, target):
        if (stimulus == target).all():
            trial_err = 1 if (stimulus != choice).any() else 0
        elif (stimulus != target).any():
            trial_err = 1 if (stimulus == choice).all() else 0
        return trial_err

    def compute_posterior(self, trial_type):
        likelihood = list(map(lambda x:
                              x if trial_type == "MATCH" else (1-x), self.ASSOCIATION_RANGE))
        posterior = (likelihood * self.prior) / np.sum(likelihood * self.prior)
        return posterior

    def update_v(self, stimulus, choice, target):
        # Waiting until distribution has converged...
        if not self.has_dist_convgered:
            if (len(self.std_history) >= self.STD_WINDOW_SZ and np.mean(self.std_history) < self.STD_THRESH):
                self.has_dist_convgered = True

            trial_type = "MATCH" if (stimulus == target).all() else "NON-MATCH"
            self.prior = self.compute_posterior(trial_type)

            self.std_history.append(self.compute_std())
            if len(self.std_history) > self.STD_WINDOW_SZ:
                self.std_history.pop(0)
        # Check for changes and switch
        else:
            trial_type = "MATCH" if (stimulus == target).all() else "NON-MATCH"
            self.trial_type_history.append((trial_type))

            (v1, v2) = self.get_v()

            if len(self.trial_type_history) < self.horizon_sz:
                return

            if len(self.trial_type_history) > self.horizon_sz:
                trial_type = self.trial_type_history.pop(0)
                self.prior = self.compute_posterior(trial_type)

            tt = [1 if x == "MATCH" else 0 for x in self.trial_type_history]
            delta = np.abs(np.mean(tt) - v1)
            if delta > self.switch_thresh:
                return "SWITCH"
