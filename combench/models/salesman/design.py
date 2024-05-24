import config
from combench.interfaces.design import Design
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



from combench.models.salesman.TravelingSalesman import TravelingSalesman
from combench.models.salesman import problem1

if __name__ == '__main__':

    problem = TravelingSalesman(problem1)
    design = TSDesign(None, problem)
    print(design.vector)
    objectives = design.evaluate()
    print(objectives)

