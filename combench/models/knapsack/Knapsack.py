import math
from combench.core.model import Model
import random
import os





class Knapsack(Model):

    def __init__(self, problem_formulation):
        super().__init__(problem_formulation)
        self.synergy_matrix = problem_formulation.get('synergy_matrix', [])
        self.values = problem_formulation.get('values', [])
        self.weights = problem_formulation.get('weights', [])
        self.max_weight = problem_formulation.get('max_weight', 0)
        self.num_variables = len(self.values)
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
        max_weight = max([evals[i][1] for i in range(len(evals))])
        value_norm = abs(max_value) * 1.1
        weight_norm = max_weight * 1.1
        self.problem_store['norms'] = [value_norm, weight_norm]
        self.save_problem_store()
        return [value_norm, weight_norm]

    def random_design(self):
        design = []
        for i in range(self.num_variables):
            design.append(random.randint(0, 1))
        return design

    def evaluate(self, design, normalize=True):
        total_value, total_weight, overrun = self._evaluate(design)
        if normalize is True:
            if total_value == 1e10:
                total_value = 0.0
                total_weight = 1.0
            else:
                value_norm, weight_norm = self.norms
                total_value /= value_norm
                total_weight /= weight_norm
        return total_value, total_weight, overrun


    def _evaluate(self, design):

        # Additive item values
        value = 0
        for i in range(self.num_variables):
            if design[i] == 1:
                value += self.values[i]

        # Synergy item values
        synergy_value = 0
        for i in range(self.num_variables):
            if design[i] == 1:
                for j in range(self.num_variables):
                    if design[j] == 1:
                        synergy_value += self.synergy_matrix[i][j]

        total_value = value + synergy_value

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

        return -total_value, weight, overrun










from combench.models.knapsack import problem1


if __name__ == '__main__':
    model = Knapsack(problem1)

    design = model.random_design()
    print('Random design: {}'.format(design))
    objectives = model.evaluate(design)
    print('Objectives: {}'.format(objectives))





