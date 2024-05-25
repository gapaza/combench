import numpy as np
import random

from combench.core.design import Design
from combench.ga.UnconstrainedPop import UnconstrainedPop
from combench.ga.ConstrainedPop import ConstrainedPop


class AssigningDesign(Design):
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
        profit, cost, overrun = self.problem.evaluate(self.vector)
        self.objectives = [profit, cost]
        self.feasibility_score = overrun
        if overrun > 0.0:
            self.is_feasible = False
        else:
            self.is_feasible = True
        return self.objectives

    def get_plotting_objectives(self):
        return [-self.objectives[0], self.objectives[1]]


class AssigningPop(UnconstrainedPop):

    def __init__(self, pop_size, ref_point, problem):
        super().__init__(pop_size, ref_point, problem)

    def create_design(self, vector=None):
        design = AssigningDesign(vector, self.problem)
        return design



from combench.models.assigning import problem1
from combench.ga.NSGA2 import NSGA2
from combench.models.assigning.GeneralizedAssigning import GeneralAssigning


if __name__ == '__main__':

    # Problem
    problem = GeneralAssigning(problem1)

    # Population
    pop_size = 30
    ref_point = np.array([0, 1])
    pop = AssigningPop(pop_size, ref_point, problem)

    # NSGA2
    max_nfe = 1000
    nsga2 = NSGA2(pop, problem, max_nfe, run_name='assigning_unconstrained')
    nsga2.run()












