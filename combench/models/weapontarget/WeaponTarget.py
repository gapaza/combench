import numpy as np
from combench.interfaces.model import Model
import random



class WeaponTarget(Model):

    def __init__(self, problem_formulation):
        super().__init__(problem_formulation)
        self.num_targets = len(problem_formulation['V'])
        self.num_weapons = sum(problem_formulation['W'])
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
            evals.append(objs)
        max_target_survival = max([evals[i][0] for i in range(len(evals))])
        max_cost = max([evals[i][1] for i in range(len(evals))])
        target_survival_norm = max_target_survival * 1.1  # Add margin
        cost_norm = max_cost * 1.1  # Add margin
        self.problem_store['norms'] = [target_survival_norm, cost_norm]
        self.save_problem_store()
        return [target_survival_norm, cost_norm]

    def random_design(self):
        num_weapons = sum(self.problem_formulation['W'])
        num_targets = len(self.problem_formulation['V'])
        design = []
        for _ in range(num_weapons):
            weapon_target = random.randint(0, num_targets-1)
            design.append(weapon_target)
        return design

    def evaluate(self, design, normalize=True):
        target_survival, cost = self._evaluate(design)
        if normalize is True:
            target_survival_norm, cost_norm = self.norms
            target_survival /= target_survival_norm
            cost /= cost_norm
        return target_survival, cost

    def _evaluate(self, design):
        # Extract problem formulation parameters
        V = self.problem_formulation['V']  # Target values
        W = self.problem_formulation['W']  # Number of weapons of each type
        p = self.problem_formulation['p']  # Probability of destroying target j by weapon of type i
        C = self.problem_formulation['C']  # Cost of each weapon type

        # Create list that encodes weapon to type
        weapon_types = []
        weapon_costs = []
        for idx, num_weapons in enumerate(W):
            weapon_types.extend([idx] * num_weapons)
            weapon_costs.extend([C[idx]] * num_weapons)
        num_targets = len(V)

        survival_probabilities = [1 for _ in range(num_targets)]  # Survival probability for each target
        target_costs = [0 for _ in range(num_targets)]  # Cost of weapon deployment per-target
        weapons_per_target = [0 for _ in range(num_targets)]  # Number of weapons deployed per-target
        for idx, bit in enumerate(design):
            target_idx = bit
            weapon_type = weapon_types[idx]
            prob_destroy = p[weapon_type][target_idx]
            prob_survive = 1 - prob_destroy
            survival_probabilities[target_idx] *= prob_survive

            weapon_cost = weapon_costs[idx]
            target_costs[target_idx] += weapon_cost
            weapons_per_target[target_idx] += 1

        # Calculate the value estimate
        value_estimates = []  # Minimize
        for idx, survival_prob in enumerate(survival_probabilities):
            value_estimates.append(V[idx] * survival_prob)
        value_estimate = sum(value_estimates)

        # Calculate the cost estimate
        cost_estimates = []
        for idx, target_cost in enumerate(target_costs):
            if target_cost == 0:
                continue
            target_weapon_count = weapons_per_target[idx]
            discount = 1.0 - (target_weapon_count * 0.05)  # 5% discount per weapon deployed
            discounted_target_cost = target_cost * discount
            cost_estimates.append(discounted_target_cost)
        cost_estimate = sum(cost_estimates)

        return value_estimate, cost_estimate


from combench.models.weapontarget import problem1

if __name__ == '__main__':

    design = [0, 1, 3, 3, 4, 5, 6]

    wt = WeaponTarget(problem1)
    result = wt.evaluate(design)
    print("Total expected survival value:", result)


