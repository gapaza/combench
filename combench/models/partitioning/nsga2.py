import numpy as np
import random

from combench.interfaces.design import Design
from combench.ga.UnconstrainedPop import UnconstrainedPop


class PartitioningDesign(Design):
    def __init__(self, vector, problem):
        super().__init__(vector, problem)

    def random_design(self):
        return self.problem.random_design()

    def mutate(self):
        prob_mutate = 1.0 / self.num_vars
        num_sets = len(set(self.vector))
        for i in range(self.num_vars):
            if random.random() < prob_mutate:
                self.vector[i] = random.randint(0, num_sets)

    def evaluate(self):
        if self.is_evaluated():
            return self.objectives
        synergy, weight, overweight = self.problem.evaluate(self.vector)
        self.objectives = [synergy, weight]
        self.feasibility_score = overweight
        self.is_feasible = True  # Assume all overweight designs are feasible for now
        return self.objectives

    def get_plotting_objectives(self):
        return [-self.objectives[0], self.objectives[1]]


class PartitioningPopulation(UnconstrainedPop):

    def __init__(self, pop_size, ref_point, problem):
        super().__init__(pop_size, ref_point, problem)

    def create_design(self, vector=None):
        design = PartitioningDesign(vector, self.problem)
        return design



from combench.models.partitioning import problem1
from combench.ga.NSGA2 import NSGA2
from combench.models.partitioning.SetPartitioning import SetPartitioning


if __name__ == '__main__':

    # Problem
    problem = SetPartitioning(problem1)

    # Population
    pop_size = 30
    ref_point = np.array([0, 1])
    pop = PartitioningPopulation(pop_size, ref_point, problem)

    # NSGA2
    max_nfe = 10000
    nsga2 = NSGA2(pop, problem, max_nfe, run_name='set-partitioning')
    nsga2.run()












