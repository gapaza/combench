import numpy as np
import math
from combench.core.model import Model
from combench.models.utils import random_binary_design, random_binary_design2
import time

class GeneralAssigning(Model):

    def __init__(self, problem_formulation):
        super().__init__(problem_formulation)
        self.num_agents = problem_formulation['num_agents']
        self.num_tasks = problem_formulation['num_tasks']
        self.costs = problem_formulation['costs']
        self.profits = problem_formulation['profits']
        self.budgets = problem_formulation['budgets']
        self.norms = self.load_norms()
        print('Norm values: {}'.format(self.norms))



    def load_norms(self):
        if 'norms' in self.problem_store:
            return self.problem_store['norms']
        print('Calculating new norms\n\n\n')
        time.sleep(2)

        # Calculate the norms
        random_designs = [self.random_design() for _ in range(10000)]

        evals = []
        for design in random_designs:
            objs = self.evaluate(design, normalize=False)
            if objs[1] == 1e10:
                continue
            evals.append(objs)

        max_profit = min([evals[i][0] for i in range(len(evals))])
        max_cost = max([evals[i][1] for i in range(len(evals))])
        profit_norm = abs(max_profit) * 1.1
        cost_norm = max_cost * 1.1
        self.problem_store['norms'] = [profit_norm, cost_norm]
        self.save_problem_store()
        return [profit_norm, cost_norm]


    def random_design(self):
        return random_binary_design2(self.num_tasks * self.num_agents)
        # return random_binary_design(self.num_tasks * self.num_agents)

    def evaluate(self, design, normalize=True):
        profit, cost, overrun = self._evaluate(design)
        if normalize is True:
            if cost == 1e10:
                profit = 0.0
                cost = 1.0
            else:
                profit_norm, cost_norm = self.norms
                profit = profit / profit_norm
                cost = cost / profit_norm
        return [profit, cost, overrun]

    def _evaluate(self, design):
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
        budget_overruns = 0
        for agent in range(self.num_agents):
            total_cost += agent_costs[agent]
            if agent_costs[agent] > self.budgets[agent]:
                valid_design = False
                budget_overruns = abs(agent_costs[agent] - self.budgets[agent])

        # if not valid_design:
        #     return 0, 1e10, 1e10
        # else:
        #     return -total_profit, total_cost, budget_overruns
        return -total_profit, total_cost, budget_overruns


from combench.models.assigning import problem1

if __name__ == '__main__':

    # Example design
    design = [
        1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 1
    ]

    # Create instance and evaluate
    ga = GeneralAssigning(problem1)
    objectives = ga.evaluate(design)
    print(objectives)


