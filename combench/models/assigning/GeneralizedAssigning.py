import numpy as np
import math
from combench.interfaces.model import Model


class GeneralAssigning(Model):

    def __init__(self, problem_formulation):
        super().__init__(problem_formulation)
        self.num_agents = problem_formulation['num_agents']
        self.num_tasks = problem_formulation['num_tasks']
        self.costs = problem_formulation['costs']
        self.profits = problem_formulation['profits']
        self.budgets = problem_formulation['budgets']

    def evaluate(self, design):
        # Reshape the design to a 2D array for easier handling
        assignment_matrix = np.reshape(design, (self.num_tasks, self.num_agents))

        total_profit = 0
        agent_costs = np.zeros(self.num_agents)

        for task in range(self.num_tasks):
            for agent in range(self.num_agents):
                if assignment_matrix[task][agent] == 1:
                    agent_costs[agent] += self.costs[task][agent]
                    total_profit += self.profits[task][agent]

        # Check budget constraints
        valid_design = True
        total_cost = 0
        for agent in range(self.num_agents):
            total_cost += agent_costs[agent]
            if agent_costs[agent] > self.budgets[agent]:
                valid_design = False

        if not valid_design:
            return 0, 100
        else:
            return total_profit, total_cost


if __name__ == '__main__':
    problem_formulation = {
        'num_agents': 5,
        'num_tasks': 5,
        'costs': np.array([
            [4, 2, 5, 6, 3],
            [7, 5, 8, 6, 2],
            [3, 9, 7, 4, 8],
            [5, 4, 6, 2, 7],
            [6, 7, 5, 3, 4]
        ]),
        'profits': np.array([
            [8, 6, 7, 9, 5],
            [6, 7, 8, 5, 4],
            [5, 9, 6, 7, 8],
            [7, 6, 8, 5, 9],
            [9, 8, 7, 6, 5]
        ]),
        'budgets': np.array([15, 15, 15, 15, 15])
    }

    # Example design
    design = [
        1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 1
    ]

    # Create instance and evaluate
    ga = GeneralAssigning(problem_formulation)
    objectives = ga.evaluate(design)
    print(objectives)


