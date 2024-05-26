from tqdm import tqdm
from copy import deepcopy
import os
import config
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd

class BenchNSGA2:

    def __init__(self, problem, max_nfe, run_name='runner'):
        self.problem = problem
        self.max_nfe = max_nfe
        self.run_name = run_name
        self.nsga2_name = 'nsga2'

        # Save
        self.run_name = run_name
        self.save_dir = os.path.join(config.results_dir, run_name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.results_file = os.path.join(self.save_dir, 'results.json')

    def load_results(self):
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r') as f:
                results = json.load(f)
        else:
            results = []
        return results

    def plot_results(self, comparison=None):
        results = self.load_results()
        dfs = []
        for alg_run in results:
            nfe, hv = zip(*alg_run)
            min_nfe, max_nfe = min(nfe), max(nfe)
            nfe_space = np.linspace(min_nfe, max_nfe, len(hv))
            hv_interp = np.linspace(nfe_space, nfe, hv)
            df = pd.DataFrame({'nfe': nfe_space, 'hv': hv_interp, 'label': self.nsga2_name})
            dfs.append(df)



        pass

    def run(self, populations):
        results = []
        for pop in populations:
            nsga2 = NSGA2(pop, self.problem, self.max_nfe, run_name=self.nsga2_name)
            result = nsga2.run()
            results.append(result)
            with open(self.results_file, 'w') as f:
                json.dump(results, f, indent=4)
            print('Results saved to {}'.format(self.results_file))
        return results


class NSGA2:


    def __init__(self, population, problem, max_nfe=1000, run_name='NSGA2'):
        self.population = population
        self.problem = problem
        self.max_nfe = max_nfe

        # Save
        self.run_name = run_name
        self.save_dir = os.path.join(config.results_dir, run_name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)


    def run(self):

        self.population.init_population()
        self.population.eval_population()
        self.population.prune()

        progress_bar = tqdm(total=self.max_nfe)
        curr_nfe = 0
        terminated = False
        while terminated is False and curr_nfe < self.max_nfe:
            curr_nfe = deepcopy(self.population.nfe)

            self.population.create_offspring()
            self.population.eval_population()

            # Prune population
            self.population.prune()

            # Log iteration
            self.population.record()
            update_delta = self.population.nfe - curr_nfe
            progress_bar.update(update_delta)
            progress_bar.set_postfix({'hv': self.population.hv[-1]})

        # Plot hv and population
        self.population.plot_population(self.save_dir)
        self.population.plot_hv(self.save_dir)
        results = [(nfe, hv) for nfe, hv in zip(self.population.nfes, self.population.hv)]
        return results















