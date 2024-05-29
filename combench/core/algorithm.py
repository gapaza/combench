from abc import ABC, abstractmethod
import random
import numpy as np
import os
import config
import matplotlib.pyplot as plt




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





class MultiTaskAlgorithm(ABC):

    def __init__(self, problems, populations, run_name, max_nfe):
        self.problems = problems
        self.populations = populations
        self.nfe = 0
        self.max_nfe = max_nfe
        self.curr_epoch = 0
        self.run_info = {}
        self.val_run = False

        # Save
        self.run_name = run_name
        self.save_dir = os.path.join(config.results_dir, run_name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    @abstractmethod
    def run(self, *args, **kwargs):
        pass

    def record(self):
        print(self.curr_epoch, end=' | ')
        for key, value in self.run_info.items():
            if isinstance(value, list):
                print("%s: %.5f" % (key, value[-1]), end=' | ')
            else:
                print("%s: %.5f" % (key, value), end=' | ')
        self.populations[-1].prune()
        self.populations[-1].record()
        if self.val_run is True:
            min_dist = self.populations[-1].get_min_distance()
            print('nfe:', self.nfe, "hv: %.5f" % self.populations[-1].hv[-1], "min_dist: %.5f" % min_dist)
        else:
            print('')

    def get_total_nfe(self):
        total_nfe = 0
        for population in self.populations:
            total_nfe += population.nfe
        return total_nfe


    def plot_metrics(self, metrics, sn=None):
        if sn is None:
            metrics_file = os.path.join(self.save_dir, 'metrics.png')
        else:
            metrics_file = os.path.join(self.save_dir, 'metrics-'+str(sn)+'.png')

        plot_dict = {}
        for metric in metrics:
            if metric in self.run_info:
                plot_dict[metric] = self.run_info[metric]

        num_plots = len(plot_dict)
        # Define number of columns
        num_columns = 3
        # Calculate number of rows needed
        num_rows = (num_plots + num_columns - 1) // num_columns

        # Create a figure with the calculated number of rows and columns
        fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, num_rows * 3))

        # Flatten the axes array for easy iteration
        axes = axes.flatten()

        # Plot each list of floats in a subplot
        for idx, (key, values) in enumerate(plot_dict.items()):
            ax = axes[idx]
            ax.plot(values)
            ax.set_title(f'{key}')

        # Turn off unused axes
        for idx in range(num_plots, len(axes)):
            axes[idx].set_visible(False)

        # Adjust layout for better spacing
        plt.tight_layout()
        plt.savefig(metrics_file)









