import config
import json
import numpy as np

from combench.ga.UnconstrainedPop import UnconstrainedPop
from combench.models.partitioning.design import PartitioningDesign as Design






class PartitioningPopulation(UnconstrainedPop):

    def __init__(self, pop_size, ref_point, problem):
        super().__init__(pop_size, ref_point, problem)

    def create_design(self, vector=None):
        design = Design(vector, self.problem)
        return design






from combench.models.partitioning import problem1
from combench.models.partitioning.SetPartitioning import SetPartitioning


if __name__ == '__main__':

    problem = SetPartitioning(problem1)

    pop_size = 30
    ref_point = np.array([0, 1])

    pop = PartitioningPopulation(pop_size, ref_point, problem)

    pop.init_population()
    pop.eval_population()


    print(pop.calc_hv())

    pop.create_offspring()
    pop.eval_population()
    pop.prune()

    print(pop.calc_hv())







