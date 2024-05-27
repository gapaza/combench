from combench.core.population import Population
import numpy as np

from combench.core.design import Design
from combench.ga import utils as ga_utils


class UnconstrainedPop(Population):
    def __init__(self, pop_size, ref_point, problem):
        super().__init__(pop_size, ref_point, problem)

    def create_design(self, vector=None):
        return Design(vector, self.problem)

    def calc_hv(self):
        # objectives = self.eval_population()
        objectives = [design.objectives for design in self.unique_designs]
        if len(objectives) == 0:
            return 0.0
        F = np.array(objectives)
        hv = self.hv_client.do(F)
        # print(hv, objectives)
        return hv

    def prune(self):
        objectives = self.eval_population()
        F = np.array(objectives)
        fronts = self.nds.do(F, n_stop_if_ranked=self.pop_size)
        survivors = []
        exit_loop = False
        for k, front in enumerate(fronts, start=1):
            crowding_of_front = ga_utils.calc_crowding_distance(F[front, :])
            for i, idx in enumerate(front):
                design = self.designs[idx]
                design.rank = k
                design.crowding_distance = crowding_of_front[i]
                survivors.append(idx)
                if len(survivors) >= self.pop_size and k > 1:
                    exit_loop = True
                    break
            if exit_loop is True:
                break

        # Create new population
        new_population = []
        for idx in survivors:
            new_population.append(self.designs[idx])
        if len(new_population) > self.pop_size:
            new_population.sort(key=lambda x: x.crowding_distance, reverse=True)
            new_population = new_population[:self.pop_size]

        self.designs = new_population

    def create_offspring(self):

        # Set pareto rank and crowding distance
        objectives = self.eval_population()
        F = np.array(objectives)
        fronts = self.nds.do(F, n_stop_if_ranked=self.pop_size)
        for k, front in enumerate(fronts, start=1):
            crowding_of_front = ga_utils.calc_crowding_distance(F[front, :])
            for i, idx in enumerate(front):
                self.designs[idx].crowding_distance = crowding_of_front[i]
                self.designs[idx].rank = k

        # Get parent pairs
        pairs = []
        while len(pairs) < self.n_offspring:
            parent1_idx = self.binary_tournament()
            parent2_idx = self.binary_tournament()
            while parent2_idx == parent1_idx:
                parent2_idx = self.binary_tournament()
            pairs.append([
                self.designs[parent1_idx],
                self.designs[parent2_idx]
            ])

        # Create offspring
        offspring = []
        for pair in pairs:
            parent1 = pair[0]
            parent2 = pair[1]
            child = self.create_design()
            child.crossover(parent1, parent2)
            child.mutate()
            offspring.append(child)
        self.designs.extend(offspring)


from combench.models.salesman import problem1
from combench.models.salesman.TravelingSalesman import TravelingSalesman

if __name__ == '__main__':

    problem = TravelingSalesman(problem1)

    pop_size = 30
    ref_point = np.array([5, 5])

    pop = UnconstrainedPop(pop_size, ref_point, problem)







