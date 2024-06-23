from abc import ABC, abstractmethod


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






