from abc import ABC, abstractmethod
import numpy as np
import random
import math
import pint





class AbstractProblem(ABC):

    def __init__(self):
        self.ureg = pint.UnitRegistry()

    @staticmethod
    def random_problem(**kwargs):
        pass

    @staticmethod
    def get_problem(**kwargs):
        pass


    @staticmethod
    def get_mesh(x_range, x_res, y_range, y_res):
        x = np.linspace(0, x_range, x_res)
        y = np.linspace(0, y_range, y_res)
        nodes = []
        for i in x:
            for j in y:
                nodes.append([i, j])
        nodes = sorted(nodes, key=lambda x: (x[0], x[1]))
        return nodes



























