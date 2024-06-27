import numpy as np
import random
import math
from combench.models import truss

from combench.models.truss.problems.cantilever import Cantilever

def run():
    load_cond = [ # 3x3
        [0, 0], [0, 0], [0, 0],
        [0, -1], [0, 0], [0, 0],
        [0, 0], [0, 0], [0, 0],
    ]


    # 1. Get node mesh and edge nodes
    problem = Cantilever.get_problem(
        [load_cond],
        x_range=3,
        y_range=3,
        x_res=3,
        y_res=3,
        radii=0.2,
        y_modulus=210e9
    )


    design = [1 for _ in range(truss.rep.get_num_bits(problem))]
    design = [
        (0, 3),
        (1, 3),
        # (3, 4),
        # (1, 4)
    ]
    truss.rep.viz(problem, design, f_name='validation_1.png')





if __name__ == "__main__":
    run()


