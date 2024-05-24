import config
import json
import numpy as np
from copy import deepcopy

from combench.ga.UnconstrainedPop import UnconstrainedPop
from combench.models.gnc.design import GncDesign as Design






class GncPopulation(UnconstrainedPop):

    def __init__(self, pop_size, ref_point, problem):
        super().__init__(pop_size, ref_point, problem)

    def create_design(self, vector=None):
        design = Design(vector, self.problem)
        return design


    def eval_population(self):

        # Evaluate unknown designs
        unkonwn_designs = [design for design in self.designs if not design.is_evaluated()]
        unknown_designs_vectors = [design.vector for design in unkonwn_designs]
        unknown_designs_objectives = self.problem.evaluate_batch(unknown_designs_vectors, normalize=True)
        for design, objs in zip(unkonwn_designs, unknown_designs_objectives):
            design.objectives = objs
            design.is_feasible = True

        # Collect objectives
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






from combench.models.gnc import problem1
from combench.models.gnc.GncModel import GncModel


if __name__ == '__main__':

    problem = GncModel(problem1)

    pop_size = 30
    ref_point = np.array([0, 1])

    pop = GncPopulation(pop_size, ref_point, problem)

    pop.init_population()
    pop.eval_population()


    print(pop.calc_hv())

    pop.create_offspring()
    pop.eval_population()
    pop.prune()

    print(pop.calc_hv())







