import config
from combench.interfaces.design import Design
import random

class GncDesign(Design):
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
        if self.is_evaluated():
            return self.objectives
        reliability, mass = self.problem.evaluate(self.vector)
        self.objectives = [reliability, mass]
        self.is_feasible = True  # Assume all overweight designs are feasible for now
        return self.objectives

    def get_plotting_objectives(self):
        return [-self.objectives[0], self.objectives[1]]



from combench.models.gnc import problem1
from combench.models.gnc.GncModel import GncModel

if __name__ == '__main__':

    problem = GncModel(problem1)
    design = GncDesign(None, problem)
    print(design.vector)
    objectives = design.evaluate()
    print(objectives)

