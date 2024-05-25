from abc import ABC, abstractmethod
import random
import numpy as np
import os
import config



class Algorithm(ABC):

    def __init__(self, problem, population, run_name, max_nfe):
        self.problem = problem
        self.population = population
        self.nfe = 0
        self.max_nfe = max_nfe
        self.run_info = {

        }

        # Save
        self.run_name = run_name
        self.save_dir = os.path.join(config.results_dir, run_name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    @abstractmethod
    def run(self, *args, **kwargs):
        pass

    def record(self):
        for key, value in self.run_info.items():
            if isinstance(value, list):
                print("%s: %.5f" % (key, value[-1]), end=' | ')
            else:
                print("%s: %.5f" % (key, value), end=' | ')
        self.population.record()
        print('nfe:', self.population.nfes[-1], 'hv:', self.population.hv[-1])









