from abc import ABC, abstractmethod
import random
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.selection.tournament import compare
import json
from copy import deepcopy



class Population(ABC):

    def __init__(self, pop_size, ref_point, problem):
        self.pop_size = pop_size
        self.n_offspring = pop_size
        self.designs = []
        self.ref_point = ref_point
        self.hv_client = HV(ref_point)
        self.nds = NonDominatedSorting()
        self.problem = problem
        self.nfe = 0

        # Saved designs and evals
        self.unique_designs_bitstr = set()
        self.unique_designs = []

    def init_population(self, *args, **kwargs):
        self.designs = []
        for i in range(self.pop_size):
            design = self.create_design()
            self.designs.append(design)

    def save_population(self, file_path):
        save_data = [design.get_design_json() for design in self.designs]
        with open(file_path, 'w') as f:
            json.dump(save_data, f)

    def add_designs(self, designs):
        self.designs.extend(designs)

    def eval_population(self):
        objectives = []
        for design in self.designs:
            objs = design.evaluate()
            design_str = design.get_design_str()
            if design_str not in self.unique_designs_bitstr:
                self.unique_designs_bitstr.add(design_str)
                self.unique_designs.append(deepcopy(design))
                self.nfe += 1
            objectives.append(objs)
        return objectives

    def binary_tournament(self):
        solutions = self.designs

        select_size = len(solutions)
        if select_size > self.pop_size:
            select_size = self.pop_size

        p1 = random.randrange(select_size)
        p2 = random.randrange(select_size)
        while p1 == p2:
            p2 = random.randrange(select_size)

        player1 = solutions[p1]
        player2 = solutions[p2]

        winner_idx = compare(
            p1, player1.rank,
            p2, player2.rank,
            'smaller_is_better',
            return_random_if_equal=False
        )
        if winner_idx is None:
            winner_idx = compare(
                p1, player1.crowding_distance,
                p2, player2.crowding_distance,
                'larger_is_better',
                return_random_if_equal=True
            )
        return winner_idx

    @abstractmethod
    def calc_hv(self, *args, **kwargs):
        pass

    @abstractmethod
    def prune(self, *args, **kwargs):
        pass

    @abstractmethod
    def create_offspring(self, *args, **kwargs):
        pass

    @abstractmethod
    def create_design(self, *args, **kwargs):
        pass

