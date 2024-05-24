from combench.ga.NSGA2 import NSGA2
import numpy as np
from combench.models.partitioning import problem1
from combench.models.partitioning.SetPartitioning import SetPartitioning
from combench.models.partitioning.population import PartitioningPopulation


if __name__ == '__main__':

    # Problem
    problem = SetPartitioning(problem1)

    # Population
    pop_size = 30
    ref_point = np.array([0, 1])
    pop = PartitioningPopulation(pop_size, ref_point, problem)

    # NSGA2
    max_nfe = 10000
    nsga2 = NSGA2(pop, problem, max_nfe, run_name='SetPartitioning')
    nsga2.run()












