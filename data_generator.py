import numpy as np

class data_generator():
    def __init__(self, config):
        self.association_levels = np.array(['90', '70', '50', '30', '10'])
        self.match_trial_probability={'90':0.9, '70':.7, '50':.5, '30':.3, '10':.1}
        
        self.block_schedule= config.block_schedule
        self.ofc_control_schedule= config.ofc_control_schedule
        
        self.strategy_schedule = ['match' if bs in ['90', '70', '50'] else 'non-match' for bs in self.block_schedule]
        

    def trial_generator(self, association_level):
        '''
        generates trials given association_level in current block
        input:
        association level as one of ['30', '70', '10', '50', '90']
        returns: a tuple (input, output) e.g. in: up, out: up  ([1,0], [1,0])
        '''
        prob = self.match_trial_probability[association_level]
        #get Input: random up or down cue
        inp = np.random.choice(np.array([1., 0.])) 
        # get Output: match or non-match
        out = inp if np.random.uniform() < prob else (1 - inp)
        return (np.array([inp, 1-inp]), np.array([out, 1-out]))

    def block_generator(self,blocki):
        if blocki < len(self.block_schedule):
            # print(f'block {blocki}, association: {self.block_schedule[blocki]}')
            yield (self.block_schedule[blocki], self.ofc_control_schedule[blocki])
        else:
            print(f'Generating random blocks: {blocki}')
            yield (np.random.choice(self.association_levels), 'off')
            

class data_generator_deprecated():
    def __init__(self, local_Ntrain):
        # self.non_matches = { #cause a non-match (1.) every so many matches.
        # '90': np.array([0. if (i+1)%10!=0 else 1. for i in range(local_Ntrain) ]),
        # '70': np.array([0. if (i+1)%4!=0  else 1. for i in range(local_Ntrain)  ]),
        # '50': np.array([0. if (i+1)%2!=0  else 1. for i in range(local_Ntrain)  ]),
        # '20': np.array([1. if (i+1)%4!=0  else 0. for i in range(local_Ntrain)  ]),
        # '10': np.array([1. if (i+1)%10!=0 else 0. for i in range(local_Ntrain) ]),
        #  }

        self.non_matches = { # randomly sample non-matches (1.) with set probabilities
        '90': np.array([0. if np.random.rand()<0.9 else 1. for i in range(local_Ntrain) ]),
        '70': np.array([0. if np.random.rand()<0.7 else 1. for i in range(local_Ntrain)  ]),
        '50': np.array([0. if np.random.rand()<0.5 else 1. for i in range(local_Ntrain)  ]),
        '20': np.array([1. if np.random.rand()<0.7 else 0. for i in range(local_Ntrain)  ]),
        '10': np.array([1. if np.random.rand()<0.9 else 0. for i in range(local_Ntrain) ]),
         }
        # Trick the model by giving other associations levels
        # self.non_matches = { # randomly sample non-matches (1.) with set probabilities
        # '90': np.array([0. if np.random.rand()<0.9 else 1. for i in range(1500) ] + [0. if np.random.rand()<0.7 else 1. for i in range(500)  ] + [0. if np.random.rand()<0.9 else 1. for i in range(local_Ntrain-2000) ]),
        # '70': np.array([0. if np.random.rand()<0.7 else 1. for i in range(local_Ntrain)  ]),
        # '50': np.array([0. if np.random.rand()<0.5 else 1. for i in range(local_Ntrain)  ]),
        # '20': np.array([1. if np.random.rand()<0.7 else 0. for i in range(local_Ntrain)  ]),
        # '10': np.array([1. if np.random.rand()<0.9 else 0. for i in range(2000) ] + [0. if np.random.rand()<0.5 else 1. for i in range(500)  ] + [1. if np.random.rand()<0.9 else 0. for i in range(local_Ntrain- 2500) ]),
        #  }

        self.task_data_gen = {
        0: self.trial_generator(self.non_matches['90']),
        1: self.trial_generator(self.non_matches['10']),
        2: self.trial_generator(self.non_matches['50']),
        3: self.trial_generator(self.non_matches['20']),
        4: self.trial_generator(self.non_matches['70']),
        }

        self.training_schedule = [0, 1] *2
        self.training_schedule.append(0)
        self.training_schedule.extend([4, 1, 3, 2])

        self.training_schedule_gen = self.training_schedule_generator(self.training_schedule)

    def trial_generator(self, non_matches):
        for non_match in non_matches:
            yield (non_match)
    def training_schedule_generator(self, training_schedule):
        for contexti in training_schedule:
            yield (contexti)
                        
