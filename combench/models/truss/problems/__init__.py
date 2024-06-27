from combench.models.truss import representation as rep
import random

################################################
# ----------------- PROBLEMS ----------------- #
################################################

from combench.models.truss.problems.cantilever import Cantilever
from combench.models.truss.problems.truss_type_1 import TrussType1



# ------------------------------------------------
# 1. Variable Length Truss
# ------------------------------------------------

problem_set = TrussType1.enumerate_res({
    'x_range': 4,
    'y_range': 4,
    'x_res_range': [2, 4],
    'y_res_range': [2, 4],
    'radii': 0.2,
    'y_modulus': 210e9
})
random.seed(22)
problem_set = random.sample(problem_set, 256)
val_problem_indices = [0, 1, 2, 3]
train_problems = [problem_set[i] for i in range(len(problem_set)) if i not in val_problem_indices]
val_problems = [problem_set[i] for i in val_problem_indices]
train_problem = train_problems[0]
val_problem = val_problems[0]




