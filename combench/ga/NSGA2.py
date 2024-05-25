from tqdm import tqdm
from copy import deepcopy
import os
import matplotlib.pyplot as plt
import json


import config


class NSGA2:


    def __init__(self, population, problem, max_nfe=1000, run_name='NSGA2'):
        self.population = population
        self.problem = problem
        self.max_nfe = max_nfe

        # Metrics
        self.hv = []
        self.nfe = []

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
            self.record()
            update_delta = self.population.nfe - curr_nfe
            progress_bar.update(update_delta)

        pop_save_file = os.path.join(self.save_dir, 'population.json')
        self.population.save_population(pop_save_file)
        self.plot_run()

        results = [(nfe, hv) for nfe, hv in zip(self.nfe, self.hv)]
        with open(os.path.join(self.save_dir, 'hv_progress.json'), 'w') as f:
            json.dump(results, f)
        return results


    def record(self):
        self.population.record()

    def plot_run(self):

        # 1. Plot HV
        hv_plot_path = os.path.join(self.save_dir, 'population_hv.png')
        self.population.plot_hv(hv_plot_path)

        # 2. Plot pareto designs
        x_vals, y_vals = [], []
        for design in self.population.designs:
            if design.is_feasible is False:
                continue
            plot_objs = design.get_plotting_objectives()
            x_vals.append(plot_objs[0])
            y_vals.append(plot_objs[1])
        plt.figure(figsize=(8, 8))
        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        plt.scatter(x_vals, y_vals, c='b', label='Feasible Designs')
        plt.title('Pareto Designs')
        design_plot_file = os.path.join(self.save_dir, 'pareto_plot.png')
        plt.savefig(design_plot_file)
        plt.close('all')

        # 3. Plot all designs
        x_vals, y_vals, colors = [], [], []
        for design in self.population.unique_designs:
            designs_vals = design.get_plotting_objectives()
            x_vals.append(designs_vals[0])
            y_vals.append(designs_vals[1])
            if design.is_feasible:
                colors.append('b')
            else:
                colors.append('r')



        plt.figure(figsize=(8, 8))
        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')

        # First plot infeasible designs
        plt.scatter([x for x, c in zip(x_vals, colors) if c == 'r'],
                    [y for y, c in zip(y_vals, colors) if c == 'r'],
                    c='r', label='Infeasible Designs')
        # Then plot feasible designs
        plt.scatter([x for x, c in zip(x_vals, colors) if c == 'b'],
                    [y for y, c in zip(y_vals, colors) if c == 'b'],
                    c='b', label='Feasible Designs')


        plt.title('All Designs')
        design_plot_file = os.path.join(self.save_dir, 'all_designs_plot.png')
        plt.savefig(design_plot_file)
        plt.close('all')


















