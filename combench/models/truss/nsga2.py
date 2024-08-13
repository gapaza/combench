import config
import json
import numpy as np
import random
from copy import deepcopy
import os
import pandas as pd
import matplotlib.pyplot as plt

from combench.ga.NSGA2 import NSGA2
from combench.core.design import Design
from combench.ga.UnconstrainedPop import UnconstrainedPop
from combench.models import truss
from combench.models.truss.eval_process import EvaluationProcessManager
from combench.models.truss import plotting
from combench.models.truss.TrussModel import TrussModel


#       _____            _
#      |  __ \          (_)
#      | |  | | ___  ___ _  __ _ _ __
#      | |  | |/ _ \/ __| |/ _` | '_ \
#      | |__| |  __/\__ \ | (_| | | | |
#      |_____/ \___||___/_|\__, |_| |_|
#                           __/ |
#                          |___/


class TrussDesign(Design):
    def __init__(self, vector, problem):
        super().__init__(vector, problem)

    def random_design(self):
        return self.problem.random_design()

    def mutate(self):
        prob_mutate = 1.0 / self.num_vars
        for i in range(self.num_vars):
            if random.random() < prob_mutate:
                if self.vector[i] == 0:
                    self.vector[i] = 1
                else:
                    self.vector[i] = 0

    def evaluate(self):
        if self.is_evaluated() is True:
            return self.objectives
        stiff, vol_frac = self.problem.evaluate(self.vector)
        self.objectives = [stiff, vol_frac]
        self.is_feasible = True
        return self.objectives

    def get_plotting_objectives(self):
        return [-self.objectives[0], self.objectives[1]]


#       _____                  _       _   _
#      |  __ \                | |     | | (_)
#      | |__) |__  _ __  _   _| | __ _| |_ _  ___  _ __
#      |  ___/ _ \| '_ \| | | | |/ _` | __| |/ _ \| '_ \
#      | |  | (_) | |_) | |_| | | (_| | |_| | (_) | | | |
#      |_|   \___/| .__/ \__,_|_|\__,_|\__|_|\___/|_| |_|
#                 | |
#                 |_|


class TrussPopulation(UnconstrainedPop):

    def __init__(self, pop_size, ref_point, problem):
        super().__init__(pop_size, ref_point, problem)
        self.eval_manager = None
        self.unique_designs_epoch = []
        self.epoch = 0

    def init_eval_mamanger(self):
        if self.eval_manager is None:
            print('INITIALIZING EVAL MANAGER --> CREATING', 24, 'PROCESSES')
            self.eval_manager = EvaluationProcessManager(num_processes=24)

    def create_design(self, vector=None):
        design = TrussDesign(vector, self.problem)
        return design

    def eval_population(self):
        self.epoch += 1

        # Evaluate unknown designs
        unkonwn_designs = [design for design in self.designs if not design.is_evaluated()]
        unknown_designs_vectors = [design.vector for design in unkonwn_designs]
        if len(unknown_designs_vectors) > 0:
            self.init_eval_mamanger()
            batch_probs = [self.problem.problem_formulation for _ in range(len(unknown_designs_vectors))]
            # unknown_designs_objectives = self.eval_manager.evaluate_store(batch_probs, unknown_designs_vectors)
            unknown_designs_objectives = self.eval_manager.evaluate(batch_probs, unknown_designs_vectors)
            for design, objs in zip(unkonwn_designs, unknown_designs_objectives):
                design.objectives = objs
                design.is_feasible = True
                design.epoch = deepcopy(self.epoch)

        # Collect objectives
        objectives = []
        for design in self.designs:
            objs = design.evaluate()
            design_str = design.get_design_str()
            if design_str not in self.unique_designs_bitstr:
                self.unique_designs_bitstr.add(design_str)
                self.unique_designs.append(design)
                self.nfe += 1
                self.unique_designs_epoch.append(deepcopy(self.epoch))
            objectives.append(objs)
        return objectives

    def plot_population(self, save_dir):
        p = self.problem.problem_formulation
        pareto_plot_file = os.path.join(save_dir, 'designs_pareto.png')
        pareto_json_file = os.path.join(save_dir, 'designs_pareto.json')
        all_plot_file = os.path.join(save_dir, 'designs_all.png')
        all_epoch_plot_file = os.path.join(save_dir, 'epochs_all.png')
        all_weights_file = os.path.join(save_dir, 'weights_all.png')

        # Pareto designs
        if len(self.designs) > 0:
            plotting.plot_pareto_designs(self.designs, pareto_plot_file, pareto_json_file)

        # Viz 3 Individual Select Pareto Designs
        if len(self.designs) > 5:
            pareto_dir = os.path.join(save_dir, 'pareto')
            if not os.path.exists(pareto_dir):
                os.makedirs(pareto_dir)
            else:
                for file in os.listdir(pareto_dir):
                    os.remove(os.path.join(pareto_dir, file))
            plotting.plot_select_designs(p, self.designs, pareto_dir)

        # All designs
        if len(self.unique_designs) > 0:
            plotting.plot_all_designs(self.unique_designs, all_plot_file)

        # Plot weight graph
        if len(self.unique_designs) > 0:
            plotting.plot_weight_graph(self.unique_designs, all_weights_file)


def run_algorithm(problem, save_dir, nfe=1000, pop_size=30):
    model = TrussModel(problem)
    ref_point = np.array([0, 1])
    pop = TrussPopulation(pop_size, ref_point, model)
    nsga2 = NSGA2(pop, model, nfe, save_dir=save_dir)
    nsga2.run()
    pop.eval_manager.shutdown()





if __name__ == '__main__':

    # from combench.models.truss import train_problems, val_problems
    # v_problem = val_problems[2]
    val_num = 8
    # for val_num in range(2, 8):
    from combench.models.truss.problems.cantilever import get_problems
    train_problems, val_problems, val_problems_out = get_problems()
    v_problem = val_problems[val_num]
    truss.set_norms(v_problem)

    # Population
    p_model = TrussModel(v_problem)
    pop_size = 30
    ref_point = np.array([0, 1])
    pop = TrussPopulation(pop_size, ref_point, p_model)
    max_nfe = 5000
    nsga2 = NSGA2(pop, p_model, max_nfe, run_name=f'ga-cantilever-val-{val_num}')
    nsga2.run()

    # Valida








