import config
from combench.interfaces.design import Design
import random

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
        synergy, weight, overweight = self.problem.evaluate(self.vector)
        self.objectives = [synergy, weight]
        self.feasibility_score = overweight
        self.is_feasible = True  # Assume all overweight designs are feasible for now
        return self.objectives

    def get_plotting_objectives(self):
        return [-self.objectives[0], self.objectives[1]]



from combench.models.partitioning import problem1
from combench.models.partitioning.SetPartitioning import SetPartitioning

if __name__ == '__main__':

    problem = SetPartitioning(problem1)
    design = PartitioningDesign(None, problem)
    print(design.vector)
    objectives = design.evaluate()
    print(objectives)

