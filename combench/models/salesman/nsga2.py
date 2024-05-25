from combench.ga.NSGA2 import NSGA2
import numpy as np
from combench.models.salesman import problem1
from combench.models.salesman.TravelingSalesman import TravelingSalesman
from combench.ga.UnconstrainedPop import UnconstrainedPop
from combench.core.design import Design
import random

class TSDesign(Design):
    def __init__(self, vector, problem):
        super().__init__(vector, problem)

    def random_design(self):
        design = []
        for i in range(self.problem.num_cities):
            city_num = random.randint(0, self.problem.num_cities - 1)
            while city_num in design:
                city_num = random.randint(0, self.problem.num_cities - 1)
            design.append(city_num)
        return design

    def mutate(self):
        prob_mutate = 1.0 / self.num_vars
        for i in range(self.num_vars):
            if random.random() < prob_mutate:
                self.vector[i] = random.randint(0, self.problem.num_cities - 1)

    def evaluate(self):
        distance, cost = self.problem.evaluate(self.vector)
        self.objectives = [distance, cost]
        self.is_feasible = True
        return self.objectives

    def get_plotting_objectives(self):
        return self.objectives


class TSPopulation(UnconstrainedPop):

    def __init__(self, pop_size, ref_point, problem):
        super().__init__(pop_size, ref_point, problem)

    def create_design(self, vector=None):
        design = TSDesign(vector, self.problem)
        return design



if __name__ == '__main__':

    # Problem
    problem = TravelingSalesman(problem1)

    # Population
    pop_size = 30
    ref_point = np.array([1, 1])
    pop = TSPopulation(pop_size, ref_point, problem)

    # NSGA2
    max_nfe = 10000
    nsga2 = NSGA2(pop, problem, max_nfe, run_name='traveling-salesman')
    nsga2.run()












