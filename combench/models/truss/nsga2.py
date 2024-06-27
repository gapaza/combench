import config
import json
import numpy as np
import random
from copy import deepcopy
import os
import matplotlib.pyplot as plt

from combench.ga.NSGA2 import NSGA2
from combench.core.design import Design
from combench.ga.UnconstrainedPop import UnconstrainedPop
from combench.models import truss
from combench.models.truss.eval_process import EvaluationProcessManager


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


class TrussPopulation(UnconstrainedPop):

    def __init__(self, pop_size, ref_point, problem):
        super().__init__(pop_size, ref_point, problem)
        self.eval_manager = None

    def init_eval_mamanger(self):
        if self.eval_manager is None:
            print('INITIALIZING EVAL MANAGER --> CREATING', 12,'PROCESSES')
            self.eval_manager = EvaluationProcessManager(num_processes=12)


    def create_design(self, vector=None):
        design = TrussDesign(vector, self.problem)
        return design

    def eval_population(self):

        # Evaluate unknown designs
        unkonwn_designs = [design for design in self.designs if not design.is_evaluated()]
        unknown_designs_vectors = [design.vector for design in unkonwn_designs]
        if len(unknown_designs_vectors) > 0:
            self.init_eval_mamanger()

            batch_probs = [self.problem.problem_formulation for _ in range(len(unknown_designs_vectors))]
            unknown_designs_objectives = self.eval_manager.evaluate(batch_probs, unknown_designs_vectors)
            # unknown_designs_objectives = self.problem.evaluate_batch(unknown_designs_vectors, normalize=True)
            for design, objs in zip(unkonwn_designs, unknown_designs_objectives):
                design.objectives = objs
                design.is_feasible = True


        # Collect objectives
        objectives = []
        for design in self.designs:
            objs = design.evaluate()
            design_str = design.get_design_str()
            if design_str not in self.unique_designs_bitstr:
                self.unique_designs_bitstr.add(design_str)
                self.unique_designs.append(design)
                self.nfe += 1
            objectives.append(objs)
        return objectives

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
            json.dump(save_obj, f, indent=4)

        # Viz 3 Individual Select Pareto Designs
        if len(self.designs) > 3:
            p = self.problem.problem_formulation
            design_info = [[design.get_plotting_objectives(), design.vector] for design in self.designs if design.objectives[0] != 0]
            design_info_s = sorted(design_info, key=lambda x: x[0][0])
            design_info_v = sorted(design_info, key=lambda x: x[0][1])
            d_first_s = design_info_s[0]
            d_last_s = design_info_s[-1]
            d_first_v = design_info_v[0]
            d_last_v = design_info_v[-1]
            truss.rep.viz(p, d_first_s[1], f_name='d_stiff_low.png', base_dir=save_dir)
            truss.rep.viz(p, d_last_s[1], f_name='d_stiff_high.png', base_dir=save_dir)
            truss.rep.viz(p, d_first_v[1], f_name='d_volfrac_low.png', base_dir=save_dir)
            truss.rep.viz(p, d_last_v[1], f_name='d_volfrac_high.png', base_dir=save_dir)
        # Viz the fully connected design
        p = self.problem.problem_formulation
        fc = [1 for x in range(truss.rep.get_num_bits(p))]
        fc = truss.rep.remove_overlapping_members(p, fc)
        truss.rep.viz(p, fc, f_name='fully_connected.png', base_dir=save_dir)


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





from combench.models.truss.TrussModel import TrussModel



def t_eval(t_problem):
    design = []
    truss.rep.viz(t_problem, design, f_name='test1-design2.png')
    exit(0)


if __name__ == '__main__':
    # from combench.models.truss.problems.truss_type_1 import TrussType1
    # N = 4
    # problem_set = TrussType1.enumerate({
    #     'x_range': N,
    #     'y_range': N,
    #     'x_res': N,
    #     'y_res': N,
    #     'radii': 0.2,
    #     'y_modulus': 210e9
    # })
    # random.seed(4)
    # problem_set = random.sample(problem_set, 64)
    # print('NUM PROBLEMS:', len(problem_set))
    # problem = problem_set[0]
    # truss.set_norms(problem)
    # v_problem = problem


    from combench.models.truss import train_problems, val_problems
    v_problem = val_problems[0]
    truss.set_norms(v_problem)







    # Population
    p_model = TrussModel(v_problem)
    pop_size = 30
    ref_point = np.array([0, 1])
    pop = TrussPopulation(pop_size, ref_point, p_model)
    max_nfe = 5000
    nsga2 = NSGA2(pop, p_model, max_nfe, run_name='cantilever-4x4-ga-50res-flex1')
    nsga2.run()





