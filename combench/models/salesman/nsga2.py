from combench.ga.NSGA2 import NSGA2
import numpy as np
from combench.models.salesman.TravelingSalesman import TravelingSalesman
from combench.ga.UnconstrainedPop import UnconstrainedPop
from combench.ga.ConstrainedPop import ConstrainedPop
from combench.core.design import Design
import random
from copy import deepcopy

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
        if self.is_evaluated():
            return self.objectives
        distance, cost, is_feasible = self.problem.evaluate(self.vector, normalize=True)
        # self.objectives = [distance, cost]
        self.objectives = [distance, 0.0]
        self.is_feasible = is_feasible
        return self.objectives

    def get_plotting_objectives(self):
        return [self.objectives[0], 0]





class TSPopulation(UnconstrainedPop):

    def __init__(self, pop_size, ref_point, problem):
        super().__init__(pop_size, ref_point, problem)

    def create_design(self, vector=None):
        design = TSDesign(vector, self.problem)
        return design

    def get_min_distance(self):
        if len(self.designs) == 0:
            return None
        else:
            return min([design.objectives[0] for design in self.designs])

    def add_design(self, design):
        design_str = design.get_design_str()
        if design_str not in self.unique_designs_bitstr:
            self.unique_designs_bitstr.add(design_str)
            design.evaluate()
            copied_design = deepcopy(design)
            self.unique_designs.append(copied_design)
            self.designs.append(copied_design)
            self.nfe += 1
        else:
            bitstrs = [d.get_design_str() for d in self.unique_designs]
            idx = bitstrs.index(design_str)
            design = self.unique_designs[idx]
            design.evaluate()
        return design



from combench.models.salesman import problem1, problem2


if __name__ == '__main__':

    # Problem
    problem = TravelingSalesman(problem1)

    # Population
    pop_size = 30
    ref_point = np.array([1, 1])
    pop = TSPopulation(pop_size, ref_point, problem)

    # NSGA2
    max_nfe = 5000
    nsga2 = NSGA2(pop, problem, max_nfe, run_name='traveling-salesman-problem1')
    nsga2.run()

    print('Min distance: {}'.format(pop.get_min_distance()))













