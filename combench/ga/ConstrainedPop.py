from combench.interfaces.population import Population
import numpy as np

from combench.interfaces.design import Design
from combench.ga import utils as ga_utils


class ConstrainedPop(Population):
    def __init__(self, pop_size, ref_point, problem):
        super().__init__(pop_size, ref_point, problem)

    def create_design(self, vector=None):
        return Design(vector, self.problem)

    def calc_hv(self):
        # objectives = self.eval_population()
        feasible_unique_designs = [design for design in self.unique_designs if design.is_feasible]
        objectives = [design.objectives for design in feasible_unique_designs]
        if len(objectives) == 0:
            return 0.0
        F = np.array(objectives)
        hv = self.hv_client.do(F)
        # print(hv, objectives)
        return hv

    def prune(self):
        objectives = self.eval_population()
        F = np.array(objectives)

        # fronts = self.nds.do(F, n_stop_if_ranked=self.pop_size)
        fronts = self.custom_dominance_sorting(objectives,
           [design.feasibility_score for design in self.designs],
           [design.is_feasible for design in self.designs]
       )

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
            # Sort new population by crowding distance
            new_population.sort(key=lambda x: x.crowding_distance, reverse=True)
            new_population = new_population[:self.pop_size]

        self.designs = new_population

    def custom_dominance_sorting(self, objectives, feasibility_score, feasible):
        """
        Sort designs into Pareto fronts with custom dominance criteria.

        Parameters:
        - objectives: List of lists, where each sublist holds a design's vertical stiffness (negative value) and volume fraction (positive value).
        - feasibility_score: List of values indicating the feasibility score of each design.
        - feasible: List of booleans indicating whether each design is feasible.

        Returns:
        - List of lists, where each list holds the indices of designs in that Pareto front.
        """
        num_designs = len(objectives)

        # Initialize dominance and front structures
        dominates = {i: set() for i in range(num_designs)}
        dominated_by = {i: set() for i in range(num_designs)}
        num_dominated = [0 for _ in range(num_designs)]
        fronts = [[] for _ in range(num_designs)]  # Worst case: each individual in its own front

        # Step 1: Determine dominance relationships
        for i in range(num_designs):
            for j in range(i + 1, num_designs):
                if self.is_dominant(i, j, objectives, feasibility_score, feasible):
                    dominates[i].add(j)
                    dominated_by[j].add(i)
                    num_dominated[j] += 1
                elif self.is_dominant(j, i, objectives, feasibility_score, feasible):
                    dominates[j].add(i)
                    dominated_by[i].add(j)
                    num_dominated[i] += 1

        # Step 2: Identify the first front
        current_front = []
        for i in range(num_designs):
            if num_dominated[i] == 0:
                current_front.append(i)
                fronts[0].append(i)

        # Step 3: Build subsequent fronts
        front_index = 0
        while current_front:
            next_front = []
            for i in current_front:
                for j in dominates[i]:
                    num_dominated[j] -= 1
                    if num_dominated[j] == 0:
                        next_front.append(j)
                        fronts[front_index + 1].append(j)
            front_index += 1
            current_front = next_front

        # Remove empty fronts
        fronts = [front for front in fronts if front]

        return fronts

    def is_dominant(self, i, j, objectives, deviations, in_stiffness_window):
        """
        Determine if design i dominates design j.

        Parameters:
        - i, j: Indices of the designs being compared.
        - objectives: List of lists of objectives.
        - deviations: List of deviation values.

        Returns:
        - True if i dominates j, False otherwise.
        """

        # Design A and Design B
        # - case: both designs are in stiffness window
        # If design A and design B are in stiffness ratio window, determine dominance by objective values
        if in_stiffness_window[i] is True and in_stiffness_window[j] is True:
            for obj_i, obj_j in zip(objectives[i], objectives[j]):
                if obj_i > obj_j:  # Since lower values are better, i dominates j if both objectives are smaller
                    return False
            return True

        # - case: one design in stiffness window one not
        # - If design A is in stiffness window and design B is not, design A dominates design B
        if in_stiffness_window[i] is True and in_stiffness_window[j] is False:
            return True
        if in_stiffness_window[i] is False and in_stiffness_window[j] is True:
            return False

        # - case: both designs are not in stiffness window
        # If design A is and design B are not in stiffness window, determine dominance by stiffness ratio delta
        if in_stiffness_window[i] is False and in_stiffness_window[j] is False:
            if deviations[i] < deviations[j]:
                return True
            elif deviations[i] > deviations[j]:
                return False
            # Break ties with objective value dominance
            for obj_i, obj_j in zip(objectives[i], objectives[j]):
                if obj_i > obj_j:  # Since lower values are better, i does not dominate j if any of i's objectives are worse
                    return False
            return True



        raise ValueError('-- NO DOMINANCE CONDITIONS WERE MET, CHECK DOMINANCE FUNC')

    def create_offspring(self):

        # Set pareto rank and crowding distance
        objectives = self.eval_population()
        F = np.array(objectives)
        # fronts = self.nds.do(F, n_stop_if_ranked=self.pop_size)
        fronts = self.custom_dominance_sorting(objectives,
           [design.feasibility_score for design in self.designs],
           [design.is_feasible for design in self.designs]
       )
        # print('Objectives:', objectives)
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

    pop = ConstrainedPop(pop_size, ref_point, problem)







