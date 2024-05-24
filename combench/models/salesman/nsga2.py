from combench.ga.NSGA2 import NSGA2
import numpy as np
from combench.models.salesman import problem1
from combench.models.salesman.TravelingSalesman import TravelingSalesman
from combench.models.salesman.population import TSPopulation


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












