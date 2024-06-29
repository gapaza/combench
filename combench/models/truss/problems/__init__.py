from combench.models.truss import representation as rep
import random

################################################
# ----------------- PROBLEMS ----------------- #
################################################

from combench.models.truss.problems.cantilever import Cantilever
from combench.models.truss.problems.truss_type_1 import TrussType1



# ------------------------------------------------
# 1. Cantilever
# ------------------------------------------------

def get_cantilever(x_range, x_res, y_range, y_res, radii, y_modulus, ss=64, seed=0):
    problem_set = Cantilever.enumerate({
        'x_range': x_range,
        'y_range': y_range,
        'x_res': x_res,
        'y_res': y_res,
        'member_radii': radii,
        'youngs_modulus': y_modulus
    })
    random.seed(seed)
    sample_size = min(ss, len(problem_set))
    problem_set = random.sample(problem_set, sample_size)
    split_idx = int(0.9 * len(problem_set))
    train_problems = problem_set[:split_idx]
    val_problems = problem_set[split_idx:]
    return train_problems, val_problems


# ------------------------------------------------
# 2. Truss Type 1
# ------------------------------------------------

def get_truss_type_1(x_range, x_res, y_range, y_res, radii, y_modulus, ss=64, seed=0):
    problem_set = TrussType1.enumerate({
        'x_range': x_range,
        'y_range': y_range,
        'x_res': x_res,
        'y_res': y_res,
        'member_radii': radii,
        'youngs_modulus': y_modulus
    })
    random.seed(seed)
    sample_size = min(ss, len(problem_set))
    problem_set = random.sample(problem_set, sample_size)
    split_idx = int(0.9 * len(problem_set))
    train_problems = problem_set[:split_idx]
    val_problems = problem_set[split_idx:]
    return train_problems, val_problems


