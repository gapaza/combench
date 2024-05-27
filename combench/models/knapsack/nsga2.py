import config
import random
import numpy as np
from combench.ga.UnconstrainedPop import UnconstrainedPop
from combench.ga.ConstrainedPop import ConstrainedPop
from combench.core.design import Design
from combench.models.knapsack.Knapsack import Knapsack
from combench.models.knapsack import problem1

class KPDesign(Design):
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
        if self.is_evaluated() is True:
            return self.objectives
        value, weight, overrun = self.problem.evaluate(self.vector)
        self.objectives = [value, weight]
        self.feasibility_score = overrun
        if overrun > 0:
            self.is_feasible = False
        else:
            self.is_feasible = True
        return self.objectives

    def get_plotting_objectives(self):
        return [-self.objectives[0], self.objectives[1]]

class KPPopulation(ConstrainedPop):

    def __init__(self, pop_size, ref_point, problem):
        super().__init__(pop_size, ref_point, problem)

    def create_design(self, vector=None):
        design = KPDesign(vector, self.problem)
        return design





from combench.ga.NSGA2 import BenchNSGA2, NSGA2


if __name__ == '__main__':

    # Problem
    problem = Knapsack(problem1)
    ref_point = np.array([0, 1])


    # Populations
    pop_size = 30
    pop_batch = []
    for x in range(30):
        pop = KPPopulation(pop_size, ref_point, problem)
        pop_batch.append(pop)

    # NSGA2
    # max_nfe = 600
    # print('Running NSGA2')
    # nsga2 = NSGA2(pop_batch[0], problem, max_nfe, run_name='knapsack')
    # nsga2.run()

    # RunnerNSGA2
    max_nfe = 600
    print('Running RunnerNSGA2')
    runner = BenchNSGA2(problem, max_nfe, run_name='knapsack-study')
    # runner.run(pop_batch)
    # runner.plot_results()


