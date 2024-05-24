import numpy as np
from combench.interfaces.model import Model



class WeaponTarget(Model):

    def __init__(self, problem_formulation):
        super().__init__(problem_formulation)

    def evaluate(self, design):
        # Extract problem formulation parameters
        V = self.problem_formulation['V']  # Target values
        W = self.problem_formulation['W']  # Number of weapons of each type
        p = self.problem_formulation['p']  # Probability of destroying target j by weapon of type i

        # Create list that encodes weapon to type
        weapon_types = []
        for idx, num_weapons in enumerate(W):
            weapon_types.extend([idx] * num_weapons)
        num_targets = len(V)

        survival_probabilities = [1 for _ in range(num_targets)]
        for idx, bit in enumerate(design):
            target_idx = bit
            weapon_type = weapon_types[idx]
            prob_destroy = p[weapon_type][target_idx]
            prob_survive = 1 - prob_destroy
            survival_probabilities[target_idx] *= prob_survive

        value_estimates = []  # Minimize
        for idx, survival_prob in enumerate(survival_probabilities):
            value_estimates.append(V[idx] * survival_prob)
        value_estimate = sum(value_estimates)

        return value_estimate




if __name__ == '__main__':
    # Example usage:
    problem_formulation = {
        'V': [10, 20, 30, 40, 50, 60, 70, 80, 90],  # Target values
        'W': [4, 3],  # Number of weapons of each type
        'p': np.array([
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  # Probabilities for weapon type 1
            [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.2, 0.3]   # Probabilities for weapon type 2
        ])
    }

    design = [0, 1, 8, 3, 4, 5, 6]

    wt = WeaponTarget(problem_formulation)
    result = wt.evaluate(design)
    print("Total expected survival value:", result)


