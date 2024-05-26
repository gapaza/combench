from combench.ga.NSGA2 import NSGA2
import numpy as np
from combench.models.gnc import problem1
from combench.models.gnc.GncModel import GncModel
from combench.core.design import Design
import random
from copy import deepcopy
from combench.ga.UnconstrainedPop import UnconstrainedPop




class GncDesign(Design):
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
        if self.is_evaluated():
            return self.objectives
        reliability, mass = self.problem.evaluate(self.vector)
        self.objectives = [reliability, mass]
        if reliability == 0.0:
            self.is_feasible = False
        else:
            self.is_feasible = True
        return self.objectives

    def get_plotting_objectives(self):
        return [-self.objectives[0], self.objectives[1]]



class GncPopulation(UnconstrainedPop):

    def __init__(self, pop_size, ref_point, problem):
        super().__init__(pop_size, ref_point, problem)

    def create_design(self, vector=None):
        design = GncDesign(vector, self.problem)
        return design


    # def eval_population(self):
    #
    #     # Evaluate unknown designs
    #     unkonwn_designs = [design for design in self.designs if not design.is_evaluated()]
    #     unknown_designs_vectors = [design.vector for design in unkonwn_designs]
    #     unknown_designs_objectives = self.problem.evaluate_batch(unknown_designs_vectors, normalize=True)
    #     for design, objs in zip(unkonwn_designs, unknown_designs_objectives):
    #         design.objectives = objs
    #         design.is_feasible = True
    #
    #     # Collect objectives
    #     objectives = []
    #     for design in self.designs:
    #         objs = design.evaluate()
    #         design_str = design.get_design_str()
    #         if design_str not in self.unique_designs_bitstr:
    #             self.unique_designs_bitstr.add(design_str)
    #             self.unique_designs.append(deepcopy(design))
    #             self.nfe += 1
    #         objectives.append(objs)
    #     return objectives




if __name__ == '__main__':

    # Problem
    problem = GncModel(problem1)

    # Population
    pop_size = 50
    ref_point = np.array([0, 1])
    pop = GncPopulation(pop_size, ref_point, problem)

    # NSGA2
    max_nfe = 200
    nsga2 = NSGA2(pop, problem, max_nfe, run_name='gnc')
    nsga2.run()













