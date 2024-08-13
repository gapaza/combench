from combench.models.truss.problems.abstract_problem import AbstractProblem
import numpy as np
from copy import deepcopy
import random
import math
import os
import json
import config
from combench.models.truss.problems.cantilever import Cantilever



problem_space_file = os.path.join(config.root_dir, 'combench', 'models', 'truss', 'studies', 'cantilever1', 'problem_space.json')


# ------------------------------------------
# Problem Space Parameters
# ------------------------------------------
x_sep = 1
y_sep = 1

x_res_range = [2, 4]
y_res_range = [2, 4]

node_dropout = 0.0
member_radii = 0.1
youngs_modulus = 219e9  # Pa

all_node_load_conds = [
    [0, -1],   # Down
    # [-1, -1],  # Down and left
    # [1, -1],   # Down and right
]


# ------------------------------------------
# Enumerate Problems
# ------------------------------------------

enum_problems = []
for x_res in range(x_res_range[0], x_res_range[1] + 1):
    for y_res in range(y_res_range[0], y_res_range[1] + 1):
        res_problems = Cantilever.type_1_enum({
            'x_range': x_sep * x_res,
            'y_range': y_sep * y_res,
            'x_res': x_res,
            'y_res': y_res,
            'member_radii': member_radii,
            'youngs_modulus': youngs_modulus
        }, dropout=node_dropout, nlc=all_node_load_conds)
        for r in res_problems:
            r['x_res'] = x_res
            r['y_res'] = y_res
        if len(res_problems) > 0:
            enum_problems.append(res_problems)
all_problems = []
for p in enum_problems:
    all_problems.extend(p)
all_node_counts = [len(p['nodes']) for p in all_problems]
print('Enumerated {} problems'.format(len(all_problems)), 'with node counts:', set(all_node_counts))



# ------------------------------------------
# Validation Problems
# ------------------------------------------
# Idea is to select validation problems such that they cover a wide range of load conditions and node configurations
# A problem template is defined by a known grid of nodes and material properties
# Each templated is enumerated by defining different load conditions
# Each load condition is defined by a load imposed on a free boundary node

validation_load_conditions = [

    # 2x2 Cantilever
    {
        'y_res': 2, 'x_res': 2,
        'load_conds': [
            [0, 0], [0, 0],
            [0, -1], [0, 0]
        ]
    },

    # 2x3 Cantilever
    {
        'y_res': 2, 'x_res': 3,
        'load_conds': [
            [0, 0], [0, 0],
            [0, 0], [0, 0],
            [0, -1], [0, 0]
        ]
    },

    # 2x4 Cantilever
    {
        'y_res': 2, 'x_res': 4,
        'load_conds': [
            [0, 0], [0, 0],
            [0, 0], [0, 0],
            [0, 0], [0, 0],
            [0, -1], [0, 0]
        ]
    },

    # 3x2 Cantilever
    {
        'y_res': 3, 'x_res': 2,
        'load_conds': [
            [0, 0], [0, 0], [0, 0],
            [0, -1], [0, 0], [0, 0]
        ]
    },

    # 3x3 Cantilever
    {
        'y_res': 3, 'x_res': 3,
        'load_conds': [
            [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0],
            [0, -1], [0, 0], [0, 0]
        ]
    },

    # 3x4 Cantilever
    {
        'y_res': 3, 'x_res': 4,
        'load_conds': [
            [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0],
            [0, -1], [0, 0], [0, 0],
        ]
    },

    # 4x2 Cantilever
    {
        'y_res': 4, 'x_res': 2,
        'load_conds': [
            [0, 0], [0, 0], [0, 0], [0, 0],
            [0, -1], [0, 0], [0, 0], [0, 0]
        ]
    },

    # 4x3 Cantilever
    {
        'y_res': 4, 'x_res': 3,
        'load_conds': [
            [0, 0], [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0], [0, 0],
            [0, -1], [0, 0], [0, 0], [0, 0]
        ]
    },

    # 4x4 Cantilever
    {
        'y_res': 4, 'x_res': 4,
        'load_conds': [
            [0, 0], [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0], [0, 0],
            [0, -1], [0, 0], [0, 0], [0, 0]
        ]
    }
]

def check_val_conds(problem, val_conds):
    if problem['x_res'] == val_conds['x_res'] and problem['y_res'] == val_conds['y_res']:
        p_lcs = json.dumps(np.array(problem['load_conds'][0]).tolist())
        v_lcs = json.dumps(np.array(val_conds['load_conds']).tolist())
        if p_lcs == v_lcs:
            return True
    return False



train_problems = []
val_problems = []
for p in all_problems:
    is_validation = False
    for v in validation_load_conditions:
        if check_val_conds(p, v):
            is_validation = True
            break
    if is_validation is True:
        val_problems.append(p)
    else:
        train_problems.append(p)

print('Train Problems:', len(train_problems))
print('Validation Problems:', len(val_problems))



# ------------------------------------------
# Out of Bounds Problems
# ------------------------------------------
# Would like to assess model's ability to generalize to problems outside the training space
# - specifically, problems with different node grid configurations: 2x5, 3x5, and 4x5 cantilevers
oob_validation_problems = []

# 2x5 Cantilever
oob_problems1 = Cantilever.type_1_enum(
    {
        'x_range': 5,
        'y_range': 2,
        'x_res': 5,
        'y_res': 2,
        'member_radii': member_radii,
        'youngs_modulus': youngs_modulus
    }
)
for p in oob_problems1:
    p['x_res'] = 5
    p['y_res'] = 2
oob_val_conds = {
    'x_res': 5, 'y_res': 2,
    'load_conds': [
        [0, 0], [0, 0],
        [0, 0], [0, 0],
        [0, 0], [0, 0],
        [0, 0], [0, 0],
        [0, -1], [0, 0],
    ]
}
oob_validation_problems.extend([p for p in oob_problems1 if check_val_conds(p, oob_val_conds)])

# 3x5 Cantilever
oob_problems2 = Cantilever.type_1_enum(
    {
        'x_range': 5,
        'y_range': 3,
        'x_res': 5,
        'y_res': 3,
        'member_radii': member_radii,
        'youngs_modulus': youngs_modulus
    }
)
for p in oob_problems2:
    p['x_res'] = 5
    p['y_res'] = 3
oob_val_conds = {
    'x_res': 5, 'y_res': 3,
    'load_conds': [
        [0, 0], [0, 0], [0, 0],
        [0, 0], [0, 0], [0, 0],
        [0, 0], [0, 0], [0, 0],
        [0, 0], [0, 0], [0, 0],
        [0, -1], [0, 0], [0, 0],
    ]
}
oob_validation_problems.extend([p for p in oob_problems2 if check_val_conds(p, oob_val_conds)])

# 4x5 Cantilever
oob_problems3 = Cantilever.type_1_enum(
    {
        'x_range': 5,
        'y_range': 4,
        'x_res': 5,
        'y_res': 4,
        'member_radii': member_radii,
        'youngs_modulus': youngs_modulus
    }
)
for p in oob_problems3:
    p['x_res'] = 5
    p['y_res'] = 4
oob_val_conds = {
    'x_res': 5, 'y_res': 4,
    'load_conds': [
        [0, 0], [0, 0], [0, 0], [0, 0],
        [0, 0], [0, 0], [0, 0], [0, 0],
        [0, 0], [0, 0], [0, 0], [0, 0],
        [0, 0], [0, 0], [0, 0], [0, 0],
        [0, -1], [0, 0], [0, 0], [0, 0],
    ]
}
oob_validation_problems.extend([p for p in oob_problems3 if check_val_conds(p, oob_val_conds)])

print('Out of Bounds Validation Problems:', len(oob_validation_problems))








