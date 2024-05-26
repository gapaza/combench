import math
from combench.core.model import Model
import random
import os





class Knapsack2(Model):

    def __init__(self, problem_formulation):
        super().__init__(problem_formulation)
        self.values_1 = problem_formulation.get('values_1', [])
        self.values_2 = problem_formulation.get('values_2', [])
        self.weights = problem_formulation.get('weights', [])
        self.max_weight = problem_formulation.get('max_weight', 0)
        self.num_variables = len(self.values_1)
        self.norms = self.load_norms()
        print('Norm values: {}'.format(self.norms))



    def load_norms(self):
        if 'norms' in self.problem_store:
            return self.problem_store['norms']

        # Calculate the norms
        random_designs = [self.random_design() for _ in range(10000)]
        evals = []
        for design in random_designs:
            objs = self.evaluate(design, normalize=False)
            if objs[0] == 1e10:
                continue
            evals.append(objs)
        max_value = min([evals[i][0] for i in range(len(evals))])
        max_weight = min([evals[i][1] for i in range(len(evals))])
        value_norm = abs(max_value) * 1.3
        weight_norm = abs(max_weight) * 1.3
        self.problem_store['norms'] = [value_norm, weight_norm]
        self.save_problem_store()
        return [value_norm, weight_norm]

    def random_design(self):
        design = []
        for i in range(self.num_variables):
            design.append(random.randint(0, 1))
        return design

    def evaluate(self, design, normalize=True):
        value_1, value_2, overrun = self._evaluate(design)
        if normalize is True:
            value_norm, weight_norm = self.norms
            value_1 /= value_norm
            value_2 /= weight_norm
        return value_1, value_2, overrun


    def _evaluate(self, design):

        # Additive item values
        value_1 = 0
        for i in range(self.num_variables):
            if design[i] == 1:
                value_1 += self.values_1[i]

        # Value 2
        value_2 = 0
        for i in range(self.num_variables):
            if design[i] == 1:
                value_2 += self.values_2[i]

        # Additive item weights
        weight = 0
        for i in range(self.num_variables):
            if design[i] == 1:
                weight += self.weights[i]

        overrun = 0
        valid_design = True
        if weight > self.max_weight:
            valid_design = False
            overrun = weight - self.max_weight

        if valid_design is False:
            value_1 = 0
            value_2 = 0

        return -value_1, -value_2, overrun










from combench.models.knapsack import problem1


if __name__ == '__main__':
    model = Knapsack2(problem1)

    design = model.random_design()
    print('Random design: {}'.format(design))
    objectives = model.evaluate(design)
    print('Objectives: {}'.format(objectives))





