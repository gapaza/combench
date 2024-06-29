from abc import ABC, abstractmethod
import random
import numpy as np
from combench.models import utils as d_utils


class Design(ABC):

    def __init__(self, vector, problem):
        self.problem = problem
        self.vector = vector  # This is the design vector
        if self.vector is None:
            self.vector = self.random_design()
        self.num_vars = len(self.vector)
        self.is_feasible = True
        self.feasibility_score = 0.0
        self.objectives = None
        self.rank = None
        self.crowding_distance = None
        self.memory = None
        self.epoch = None
        self.weight = None

    def is_evaluated(self):
        return self.objectives is not None

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def get_plotting_objectives(self, *args, **kwargs):
        pass

    @abstractmethod
    def random_design(self):
        return self.problem.random_design()

    @abstractmethod
    def mutate(self):
        pass

    def crossover(self, p1_design, p2_design, c_type='uniform'):
        if c_type == 'point':
            crossover_point = random.randint(1, self.num_vars)
            child_vector = list(np.concatenate((p1_design.vector[:crossover_point], p2_design.vector[crossover_point:])))
        elif c_type == 'uniform':
            child_vector = [random.choice([m_bit, f_bit]) for m_bit, f_bit in zip(p1_design.vector, p2_design.vector)]
        else:
            raise ValueError('Invalid crossover type: {}'.format(c_type))
        self.vector = child_vector

    def get_design_str(self):
        return ''.join([str(bit) for bit in self.vector])

    def get_design_json(self):
        return {'vector': self.get_design_str(), 'objectives': self.objectives}







