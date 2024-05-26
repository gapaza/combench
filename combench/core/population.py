from abc import ABC, abstractmethod
import random
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.selection.tournament import compare
import json
from copy import deepcopy
import os
import matplotlib.pyplot as plt



class Population(ABC):

    def __init__(self, pop_size, ref_point, problem):
        self.pop_size = pop_size
        self.n_offspring = pop_size
        self.designs = []
        self.ref_point = ref_point
        self.hv_client = HV(ref_point)
        self.nds = NonDominatedSorting()
        self.problem = problem
        self.nfe = 0

        # Saved designs and evals
        self.unique_designs_bitstr = set()
        self.unique_designs = []

        # Metrics
        self.hv = []
        self.nfes = []

    def init_population(self, *args, **kwargs):
        self.designs = []
        for i in range(self.pop_size):
            design = self.create_design()
            self.designs.append(design)

    def add_design(self, design):
        design_str = design.get_design_str()
        if design_str not in self.unique_designs_bitstr:
            self.unique_designs_bitstr.add(design_str)
            design.evaluate()
            copied_design = deepcopy(design)
            self.unique_designs.append(copied_design)
            self.designs.append(copied_design)
            self.nfe += 1
        else:
            bitstrs = [d.get_design_str() for d in self.unique_designs]
            idx = bitstrs.index(design_str)
            design = self.unique_designs[idx]
            design.evaluate()
        return design

    def eval_population(self):
        objectives = []
        for design in self.designs:
            objs = design.evaluate()
            design_str = design.get_design_str()
            if design_str not in self.unique_designs_bitstr:
                self.unique_designs_bitstr.add(design_str)
                self.unique_designs.append(deepcopy(design))
                self.nfe += 1
            objectives.append(objs)
        return objectives

    def binary_tournament(self):
        solutions = self.designs

        select_size = len(solutions)
        if select_size > self.pop_size:
            select_size = self.pop_size

        p1 = random.randrange(select_size)
        p2 = random.randrange(select_size)
        while p1 == p2:
            p2 = random.randrange(select_size)

        player1 = solutions[p1]
        player2 = solutions[p2]

        winner_idx = compare(
            p1, player1.rank,
            p2, player2.rank,
            'smaller_is_better',
            return_random_if_equal=False
        )
        if winner_idx is None:
            winner_idx = compare(
                p1, player1.crowding_distance,
                p2, player2.crowding_distance,
                'larger_is_better',
                return_random_if_equal=True
            )
        return winner_idx

    def record(self):
        self.hv.append(self.calc_hv())
        self.nfes.append(deepcopy(self.nfe))

    def plot_hv(self, save_path):
        plot_file = os.path.join(save_path, 'population_hv.png')
        json_file = os.path.join(save_path, 'population_hv.json')
        plt.figure(figsize=(8, 8))
        plt.plot(self.nfes, self.hv)
        plt.xlabel('NFE')
        plt.ylabel('HV')
        plt.title('HV Progress')
        plt.savefig(plot_file)
        plt.close('all')
        save_obj = [(nfe, hv) for nfe, hv in zip(self.nfes, self.hv)]
        with open(json_file, 'w') as f:
            json.dump(save_obj, f)

    def plot_population(self, save_dir):
        pareto_plot_file = os.path.join(save_dir, 'designs_pareto.png')
        pareto_json_file = os.path.join(save_dir, 'designs_pareto.json')
        all_plot_file = os.path.join(save_dir, 'designs_all.png')

        # Pareto designs
        x_vals_f, y_vals_f = [], []
        x_vals_i, y_vals_i = [], []
        for design in self.designs:
            objectives = design.get_plotting_objectives()
            if design.is_feasible is True:
                x_vals_f.append(objectives[0])
                y_vals_f.append(objectives[1])
            else:
                x_vals_i.append(objectives[0])
                y_vals_i.append(objectives[1])
        plt.figure(figsize=(8, 8))
        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        plt.scatter(x_vals_f, y_vals_f, c='b', label='Feasible Designs')
        plt.scatter(x_vals_i, y_vals_i, c='r', label='Infeasible Designs')
        plt.title('Pareto Front')
        plt.legend()
        plt.savefig(pareto_plot_file)
        plt.close('all')
        save_obj = [design.get_design_json() for design in self.designs if design.is_feasible]
        with open(pareto_json_file, 'w') as f:
            json.dump(save_obj, f)

        # All designs
        x_vals_f, y_vals_f = [], []
        x_vals_i, y_vals_i = [], []
        for design in self.unique_designs:
            objectives = design.get_plotting_objectives()
            if design.is_feasible is True:
                x_vals_f.append(objectives[0])
                y_vals_f.append(objectives[1])
            else:
                x_vals_i.append(objectives[0])
                y_vals_i.append(objectives[1])
        plt.figure(figsize=(8, 8))
        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        plt.scatter(x_vals_f, y_vals_f, c='b', label='Feasible Designs')
        plt.scatter(x_vals_i, y_vals_i, c='r', label='Infeasible Designs')
        plt.title('All Designs')
        plt.legend()
        plt.savefig(all_plot_file)
        plt.close('all')

    @abstractmethod
    def calc_hv(self, *args, **kwargs):
        pass

    @abstractmethod
    def prune(self, *args, **kwargs):
        pass

    @abstractmethod
    def create_offspring(self, *args, **kwargs):
        pass

    @abstractmethod
    def create_design(self, *args, **kwargs):
        pass

