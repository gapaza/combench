import config
import json
import numpy as np

from combench.ga.UnconstrainedPop import UnconstrainedPop
from combench.models.salesman.design import TSDesign as Design






class TSPopulation(UnconstrainedPop):

    def __init__(self, pop_size, ref_point, problem):
        super().__init__(pop_size, ref_point, problem)

    def create_design(self, vector=None):
        design = Design(vector, self.problem)
        return design






from combench.models.salesman import problem1
from combench.models.salesman.TravelingSalesman import TravelingSalesman


if __name__ == '__main__':

    problem = TravelingSalesman(problem1)

    pop_size = 30
    ref_point = np.array([100, 100])

    pop = TSPopulation(pop_size, ref_point, problem)

    pop.init_population()
    pop.eval_population()


    print(pop.calc_hv())

    pop.create_offspring()
    pop.eval_population()
    pop.prune()

    print(pop.calc_hv())







