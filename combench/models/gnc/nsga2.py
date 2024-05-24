from combench.ga.NSGA2 import NSGA2
import numpy as np
from combench.models.gnc import problem1
from combench.models.gnc.GncModel import GncModel
from combench.models.gnc.population import GncPopulation
from combench.models.gnc.design import GncDesign



if __name__ == '__main__':

    # Problem
    problem = GncModel(problem1)

    # Population
    pop_size = 50
    ref_point = np.array([0, 1])
    pop = GncPopulation(pop_size, ref_point, problem)

    # NSGA2
    max_nfe = 1000
    nsga2 = NSGA2(pop, problem, max_nfe, run_name='gnc')
    nsga2.run()













