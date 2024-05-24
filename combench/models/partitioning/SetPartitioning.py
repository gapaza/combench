import numpy as np
from combench.interfaces.model import Model
import random


class SetPartitioning(Model):

    def __init__(self, problem_formulation):
        super().__init__(problem_formulation)
        self.max_weight = problem_formulation.get('max_weight')
        self.synergy_matrix = np.array(problem_formulation.get('synergy_matrix'))
        self.cost_matrix = np.array(problem_formulation.get('cost_matrix'))
        self.weights = problem_formulation.get('weights')
        self.norms = self.load_norms()
        print('Norm values: {}'.format(self.norms))

    def load_norms(self):
        if 'norms' in self.problem_store:
            return self.problem_store['norms']

        # Calculate the norms
        random_designs = [self.random_design() for _ in range(10000)]
        # print(random_designs[0])
        evals = []
        for design in random_designs:
            objs = self.evaluate(design, normalize=False)
            if objs[1] == 1e10:
                continue
            evals.append(objs)
        max_synergy = min([evals[i][0] for i in range(len(evals))])
        max_weight = max([evals[i][1] for i in range(len(evals))])
        synergy_norm = abs(max_synergy) * 1.1
        weight_norm = max_weight * 1.1
        self.problem_store['norms'] = [synergy_norm, weight_norm]
        self.save_problem_store()
        return [synergy_norm, weight_norm]

    def random_design(self):
        design = []
        sets = set()
        curr_set = 1
        set_range = [1, curr_set]
        for i in range(len(self.weights)):
            val = random.randint(set_range[0], set_range[1])
            design.append(val)
            if val not in sets:
                sets.add(val)
                curr_set += 1
                set_range = [1, curr_set]
        return design



    def evaluate(self, partition, normalize=True):
        synergy, cost, excess_weight = self._evaluate(partition)
        if normalize is True:
            if cost == 1e10:
                synergy = 0.0
                cost = 1.0
            else:
                synergy_norm, weight_norm = self.norms
                synergy /= synergy_norm
                cost /= weight_norm
        return -synergy, cost, excess_weight

    def _evaluate(self, partition):
        # Verify the partitioning solution
        set_weights = {}
        set_synergy = {}
        set_costs = {}

        # Calculate the total synergy for each set
        for idx, set_id in enumerate(partition):
            if set_id not in set_weights:
                set_weights[set_id] = 0
                set_synergy[set_id] = 0
                set_costs[set_id] = 0
            set_weights[set_id] += self.weights[idx]

            # Calculate the synergy between sets
            for jdx, other_set_id in enumerate(partition):
                if set_id == other_set_id and idx != jdx:
                    set_synergy[set_id] += self.synergy_matrix[idx][jdx]
                    set_costs[set_id] += self.cost_matrix[idx][jdx]


        # Check if any set exceeds the maximum weight
        excess_weight = 0
        total_weight = 0
        valid_design = True
        for set_id, weight in set_weights.items():
            total_weight += weight
            if weight > self.max_weight:
                valid_design = False
                excess_weight += weight - self.max_weight
                set_synergy[set_id] = 0

        # Sum up the total synergy
        total_synergy = sum(set_synergy.values())

        # Sum up the total cost
        total_cost = sum(set_costs.values())

        # if valid_design is False:
        #     total_synergy = 0
        #     total_cost = 1e10


        return total_synergy, total_cost, excess_weight

from combench.models.partitioning import problem1

if __name__ == '__main__':

    partition = [1, 2, 3, 1, 4, 3, 2, 2, 1, 1]  # Example partitioning solution

    sp = SetPartitioning(problem1)
    result = sp.evaluate(partition)
    print("Result:", result)

    partition = sp.random_design()
    print("Random partitioning solution:", partition)
    result = sp.evaluate(partition)
    print("Result:", result)



