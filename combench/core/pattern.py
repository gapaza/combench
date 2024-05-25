from abc import ABC, abstractmethod


class Pattern(ABC):

    @abstractmethod
    def random(self):
        pass

    @abstractmethod
    def crossover(self, parent1, parent2):
        pass

    @abstractmethod
    def mutate(self, child):
        pass












