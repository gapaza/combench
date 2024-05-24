import config
import json
import numpy as np
import random

from combench.ga.NSGA2 import NSGA2
from combench.interfaces.design import Design
from combench.ga.UnconstrainedPop import UnconstrainedPop

class WTDesign(Design):
    def __init__(self, vector, problem):
        super().__init__(vector, problem)

    def random_design(self):
        return self.problem.random_design()

    def mutate(self):
        prob_mutate = 1.0 / self.num_vars
        for i in range(self.num_vars):
            if random.random() < prob_mutate:
                self.vector[i] = random.randint(0, self.problem.num_targets - 1)

    def evaluate(self):
        target_survival, cost = self.problem.evaluate(self.vector)
        self.objectives = [target_survival, cost]
        self.is_feasible = True
        return self.objectives

    def get_plotting_objectives(self):
        return self.objectives

class WTPopulation(UnconstrainedPop):

    def __init__(self, pop_size, ref_point, problem):
        super().__init__(pop_size, ref_point, problem)

    def create_design(self, vector=None):
        design = WTDesign(vector, self.problem)
        return design



from combench.models.weapontarget import problem1
from combench.models.weapontarget.WeaponTarget import WeaponTarget

if __name__ == '__main__':

    problem = WeaponTarget(problem1)

    # Population
    pop_size = 30
    ref_point = np.array([1, 1])
    pop = WTPopulation(pop_size, ref_point, problem)

    # NSGA2
    max_nfe = 10000
    nsga2 = NSGA2(pop, problem, max_nfe, run_name='weapon-target')
    nsga2.run()


