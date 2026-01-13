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

from tqdm import tqdm

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
        stiff, vol_frac, constraint_value = self.problem.evaluate(self.vector)
        self.objectives = [stiff, vol_frac]
        # self.is_feasible = True
        if constraint_value > 0:
            self.is_feasible = False

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
        if len(self.designs) > 0:
            feasible_dir = os.path.join(save_dir, 'feasible')
            if not os.path.exists(feasible_dir):
                os.makedirs(feasible_dir)
            else:
                for file in os.listdir(feasible_dir):
                    os.remove(os.path.join(feasible_dir, file))
            plotting.plot_feasible_designs(p, self.designs, feasible_dir)

        # # Any designs
        # if len(self.designs) > 0:
        #     any_dir = os.path.join(save_dir, 'any')
        #     if not os.path.exists(any_dir):
        #         os.makedirs(any_dir)
        #     else:
        #         for file in os.listdir(any_dir):
        #             os.remove(os.path.join(any_dir, file))
        #     plotting.plot_any_designs(p, self.designs, any_dir)



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



# -----------------------------------------------------------
# SHINKA ALGORITHM
# -----------------------------------------------------------



from typing import List, Tuple, Dict, Optional, Set, Any
import numpy as np

N_ALLOWED_FUNCTION_EVALS = 10000




# EVOLVE-BLOCK-START
import random
import math
from itertools import chain, combinations


def search_algorithm(truss_problem):
    """Search algorithm based on NodeSort"""

    problem_formulation = truss_problem.get_problem_formulation()
    nodes = problem_formulation['nodes']
    print(nodes)


    all_node_idx = [x for x in range(len(nodes))]
    fixed_node_idx = truss_problem.get_fixed_nodes()
    load_node_idx = truss_problem.get_load_nodes()

    static_nodes = fixed_node_idx + load_node_idx
    free_nodes = [x for x in all_node_idx if x not in static_nodes]
    print('static nodes: ', static_nodes)
    print('free_nodes: ', free_nodes)

    # Get all combinations of fixed nodes
    fixed_nodes_pwr_set = list(power_set(fixed_node_idx))
    static_nodes_comb_feasible = []
    for fixed_nodes_comb in fixed_nodes_pwr_set:
        if len(fixed_nodes_comb) > 1:
            static_nodes_comb_feasible.append(list(fixed_nodes_comb) + load_node_idx)

    # Get all combinations of free nodes
    free_nodes_pwr_set = list(power_set(free_nodes))

    # Iterate over all combinations of free nodes, and generate a design for each with NodeSort
    designs = []
    for static_nodes_comb in static_nodes_comb_feasible:
        for free_node_comb in tqdm(free_nodes_pwr_set):
            # node_comb = list(free_node_comb) + static_nodes
            node_comb = list(free_node_comb) + static_nodes_comb
            node_comb = list(set(node_comb))

            nodes_new = [nodes[idx] for idx in node_comb]
            # print('nodes new: ', nodes_new)

            # Get NodeSort design and convert
            edges_new = node_sort_truss(nodes_new)
            bit_list, bit_str, node_idx_pairs, node_coords = truss_problem.convert(edges_new)

            # Remove members connecting fixed nodes
            node_idx_pairs_fixed = remove_members_connecting_fixed(node_idx_pairs, fixed_node_idx)
            bit_list, bit_str, node_idx_pairs, node_coords = truss_problem.convert(node_idx_pairs_fixed)

            # Append design to design set
            designs.append(bit_list)

    return np.array(designs)


def power_set(iterable):
    """
    Returns an iterator over all subsets (the power set) of the iterable.
    The subsets are returned as tuples, including the empty tuple ().
    """
    s = list(iterable)
    # Generates combinations for all possible lengths r from 0 to len(s)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

def node_sort_truss(nodes):
    """
    Implements the NodeSort truss design algorithm with corrected
    boundary processing and triangulation logic.

    Args:
        nodes (list of tuple): A list of nodes, where each node is an (x, y) tuple.

    Returns:
        list of tuple: A list of edges representing the truss design.
                       ((x1, y1), (x2, y2))
    """
    # 1. Sort nodes ascending over x, if equal compare over y
    sorted_nodes = sorted(nodes)
    n = len(sorted_nodes)

    edges = []

    # 2. Point to first node i = 0
    # 3. Loop while i < (n_i - 1)
    # Correction: The loop must process the second-to-last node
    # to ensure the final segment is connected.
    # range(n-1) goes from 0 to n-2.
    for i in range(n - 1):
        node_i = sorted_nodes[i]

        # We start checking from the very next node
        next_node_index = i + 1
        if next_node_index >= n:
            break

        node_next = sorted_nodes[next_node_index]

        targets = []

        # Determine if we are in Case A (Next node is equal/lower) or B (Next node is higher)
        # Note: The prompt says "node_i.y >= node_{i+1}.y" for A
        if node_i[1] >= node_next[1]:
            # --- CASE A: Ascending Search ---
            above_count = 0

            for j in range(i + 1, n):
                candidate = sorted_nodes[j]

                # Check Break Criterion (a): Ordering
                # We compare current candidate with the PREVIOUSLY collected target
                # to ensure the sequence of targets is ascending over y.
                if targets:
                    prev_target = targets[-1]
                    # If candidate drops below the previous target, ordering is broken
                    if candidate[1] < prev_target[1]:
                        break

                        # Check Break Criterion (b): Count of nodes above node_i
                # We collect the node first, then check if we should stop future searches.
                is_above = candidate[1] > node_i[1]

                if is_above:
                    above_count += 1

                # Add the candidate to the truss
                targets.append(candidate)

                # If this was the second node above, we break *after* adding it.
                # This ensures we capture the closing edge of the triangle.
                if above_count == 2:
                    break

        else:
            # --- CASE B: Descending Search ---
            below_count = 0

            for j in range(i + 1, n):
                candidate = sorted_nodes[j]

                # Check Break Criterion (a): Ordering (Descending)
                if targets:
                    prev_target = targets[-1]
                    # If candidate rises above the previous target, ordering is broken
                    if candidate[1] > prev_target[1]:
                        break

                # Check Break Criterion (b): Count of nodes below node_i
                is_below = candidate[1] < node_i[1]

                if is_below:
                    below_count += 1

                targets.append(candidate)

                if below_count == 2:
                    break

        # C. Define trusses
        for target in targets:
            edges.append((node_i, target))

    return edges

def remove_members_connecting_fixed(node_idx_pairs, fixed_nodes_idx):
    node_idx_pairs_new = []
    for node_idx_pair in node_idx_pairs:
        node1, node2 = node_idx_pair
        if node1 in fixed_nodes_idx and node2 in fixed_nodes_idx:
            continue
        else:
            node_idx_pairs_new.append(node_idx_pair)
    return node_idx_pairs_new












# EVOLVE-BLOCK-END







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


    # Algorithm
    p_model = TrussModel(g_problem)

    designs = search_algorithm(p_model)
    designs = designs.tolist()

    pop_size = 300
    ref_point = np.array([0, 1])
    pop = TrussPopulation(pop_size, ref_point, p_model)
    max_nfe = 100000

    for design in designs:
        truss_design = TrussDesign(design, p_model)
        pop.add_design(truss_design)
        # print('New design:', truss_design)

    pop.prune()
    hypervolume = pop.calc_hv()
    print('Hypervolume:', hypervolume)

    save_dir = '/Users/gapaza/repos/ideal/combench/plots/shinka/alg2'
    pop.plot_population(save_dir)






    # Population
    # p_model = TrussModel(g_problem)
    # pop_size = 200
    # ref_point = np.array([0, 1])
    # pop = TrussPopulation(pop_size, ref_point, p_model)
    # max_nfe = 10000
    # nsga2 = NSGA2(pop, p_model, max_nfe, run_name=f'ga-cantilever-val-{val_num}')
    # nsga2.run()

    # save_dir = '/Users/gapaza/repos/ideal/combench/plots/nsga2/unconstrained'
    # save_dir = '/Users/gapaza/repos/ideal/combench/plots/nsga2/constrained'
    # pop.plot_population(save_dir)







