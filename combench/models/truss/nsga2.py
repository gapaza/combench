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

    def create_design(self, vector=None):
        design = TrussDesign(vector, self.problem)
        return design

    def eval_population(self):

        # Evaluate unknown designs
        unkonwn_designs = [design for design in self.designs if not design.is_evaluated()]
        unknown_designs_vectors = [design.vector for design in unkonwn_designs]
        if len(unknown_designs_vectors) > 0:
            unknown_designs_objectives = self.problem.evaluate_batch(unknown_designs_vectors, normalize=True)
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
                self.unique_designs.append(deepcopy(design))
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
            design_info = [[design.get_plotting_objectives(), design.vector] for design in self.designs]
            design_info = sorted(design_info, key=lambda x: x[0][0])
            d_first = design_info[0]
            d_middle = design_info[len(design_info)//2]
            d_last = design_info[-1]
            truss.rep.viz(p, d_first[1], f_name='d1_low_stiff.png', base_dir=save_dir)
            truss.rep.viz(p, d_middle[1], f_name='d2_mid_stiff.png', base_dir=save_dir)
            truss.rep.viz(p, d_last[1], f_name='d3_high_stiff.png', base_dir=save_dir)

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





from combench.models.truss.TrussModel2 import TrussModel2
from combench.models.truss import Cantilever


if __name__ == '__main__':
    param_dict = {
        'x_range': 3,
        'y_range': 3,
        'x_res': 3,
        'y_res': 3,
        'radii': 0.2,
        'y_modulus': 210e9
    }
    problem = Cantilever.type_1(
        param_dict
    )

    p_model = TrussModel2(problem)

    # Population
    pop_size = 30
    ref_point = np.array([0, 1])
    pop = TrussPopulation(pop_size, ref_point, p_model)

    # NSGA2
    max_nfe = 1000
    nsga2 = NSGA2(pop, p_model, max_nfe, run_name='truss2')
    nsga2.run()






