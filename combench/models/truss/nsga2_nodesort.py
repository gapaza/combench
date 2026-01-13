import config
import json
import math
from collections import deque
import numpy as np
import random
from copy import deepcopy
import os
import pandas as pd
import matplotlib.pyplot as plt

from combench.ga.NSGA2 import NSGA2
from combench.core.design import Design
from combench.ga.UnconstrainedPop import UnconstrainedPop
from combench.ga.ConstrainedPop import ConstrainedPop
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

        # This is the problem object holding nodes, nodes_dof, etc...
        self.problem_formulation = problem.problem_formulation

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
        # self.is_feasible = True

        constraint_score = self.problem.evaluate_constraints(self.vector)
        self.feasibility_score = constraint_score
        if self.feasibility_score > 0:
            self.is_feasible = False
        else:
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


class TrussPopulation(ConstrainedPop):

    def __init__(self, pop_size, ref_point, problem):
        super().__init__(pop_size, ref_point, problem)
        self.eval_manager = None
        self.unique_designs_epoch = []
        self.epoch = 0

    def init_eval_mamanger(self):
        if self.eval_manager is None:
            print('INITIALIZING EVAL MANAGER --> CREATING', 24, 'PROCESSES')
            self.eval_manager = EvaluationProcessManager(num_processes=24)

    def init_population(self, *args, **kwargs):
        self.designs = []
        for i in range(self.pop_size):
            design = self.create_design()
            self.designs.append(design)

        # truss_problem = self.problem
        # load_nodes = truss_problem.get_load_nodes()
        # fixed_nodes = truss_problem.get_fixed_nodes()
        # fixed1 = fixed_nodes[0]
        # fixed2 = fixed_nodes[1]
        # hardcoded_design = []
        # for ln in load_nodes:
        #     hardcoded_design.append([fixed1, ln])
        #     hardcoded_design.append([fixed2, ln])
        # hc_bitlist, hc_bitstr, hc_node_idx_pairs, hc_node_coord_pairs = truss_problem.convert(hardcoded_design)
        # custom_design = TrussDesign(hc_bitlist, self.problem)
        # self.designs.append(custom_design)

        from combench.models.truss.nodesort import search_algorithm
        truss_problem = self.problem
        designs = search_algorithm(truss_problem)
        designs = designs.tolist()
        for design in designs:
            truss_design = TrussDesign(design, truss_problem)
            self.designs.append(truss_design)





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
                design.objectives = [objs[0], objs[1]]
                design.feasibility_score = objs[2]
                # design.feasibility_score = 0.0
                if objs[2] > 0:
                    design.is_feasible = False
                else:
                    design.is_feasible = True

                # design.objectives = objs
                # design.is_feasible = True
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

        # Feasible select designs
        if len(self.designs) > 3:
            feasible_dir = os.path.join(save_dir, 'feasible')
            if not os.path.exists(feasible_dir):
                os.makedirs(feasible_dir)
            else:
                for file in os.listdir(feasible_dir):
                    os.remove(os.path.join(feasible_dir, file))
            plotting.plot_feasible_designs(p, self.designs, feasible_dir)

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
    val_num = 1
    # for val_num in range(2, 8):
    # from combench.models.truss.problems.cantilever import get_problems
    # train_problems, val_problems, val_problems_out = get_problems()
    # v_problem = val_problems[val_num]
    # truss.set_norms(v_problem)
    # from combench.nn.trussDecoderUMD import problem as v_problem
    from tests.models.test_cantilever import g_problem

    truss.set_norms(g_problem)

    # Population
    p_model = TrussModel(g_problem)
    pop_size = 75
    ref_point = np.array([0, 1])
    pop = TrussPopulation(pop_size, ref_point, p_model)
    max_nfe = 100000
    nsga2 = NSGA2(pop, p_model, max_nfe, run_name=f'ga-cantilever-val-{val_num}')
    nsga2.run()

    # save_dir = '/Users/gapaza/repos/ideal/combench/plots/nsga2/unconstrained'
    save_dir = '/Users/gapaza/repos/ideal/combench/plots/nsga2/constrained5'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    pop.plot_population(save_dir)







