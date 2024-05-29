import numpy as np
import random


def random_binary_design(n, max_true=None):
    bit_array = np.zeros(n, dtype=int)
    ub = n if max_true is None else max_true
    num_bits_to_flip = np.random.randint(1, ub)
    indices_to_flip = np.random.choice(n, num_bits_to_flip, replace=False)
    for index in indices_to_flip:
        bit_array[index] = 1
    return bit_array.tolist()

def random_binary_design2(n, max_true=None):
    design = [random.choice([0, 1]) for x in range(n)]
    return design


def random_integer_design(n, ub):
    return list(np.random.randint(0, ub, n))



# ----------------------------------------------
# System Architecture Patterns
# ----------------------------------------------

def random_downselecting_design(choices):
    # choices is an array of ints, where each int encodes the number of choices at that level
    design = []
    for i in range(len(choices)):
        design.append(np.random.randint(0, choices[i]))
    return design









