from combench.models.truss.problems.abstract_problem import AbstractProblem
import numpy as np
from copy import deepcopy
import random
import math
import os
import json
import config

dcl = math.sqrt(2) / 2

all_node_load_conds = [
    # [-1, 0],
    [0, -1],
    # [1, 0],
    # [0, 1],
    # [1, 1],
    [-1, -1],
    [1, -1],
    # [-1, 1],
]


def get_problems():
    params = {
        'x_range': 4,
        'y_range': 4,
        'x_res_range': [2, 4],
        'y_res_range': [2, 4],
        'radii': 0.2,
        'y_modulus': 210e9
    }
    train_problems, val_problems, val_problems_out = Cantilever.enumerate(
        params,
        p_type='type2',
        dropout=0.0,
        seed=0,
        sample_size=4
    )
    print('\n\n---------------------------- Problem Set')
    print('--- Train Problems:', len(train_problems))
    print('----- Val Problems:', len(val_problems))
    print('- Val Problems Out:', len(val_problems_out))
    return train_problems, val_problems, val_problems_out


class Cantilever(AbstractProblem):


    def __init__(self):
        super().__init__()

    @staticmethod
    def enumerate(params, p_type='type1', dropout=0.0, seed=0, sample_size=64):
        random.seed(seed)
        if p_type == 'type1':
            return Cantilever.type_1_enum(params, dropout=dropout, split=True)
        elif p_type == 'type2':
            return Cantilever.type_2_enum(params, dropout=dropout, sample_size=sample_size)
        else:
            raise ValueError('Invalid problem type')

    # ----------------------------------------------------------
    # Type 1
    #   Fixed Parameters
    #   - node mesh
    #   - x node count
    #   - y node count
    #   - member radii
    #   - young's modulus
    #   - fixed nodes (left side)
    #
    #   Enumerated Parameters
    #   - loaded node locations
    # ----------------------------------------------------------

    @staticmethod
    def type_1_enum(params, dropout=0.0, split=False):
        x_range = params['x_range']
        y_range = params['y_range']
        x_res = params['x_res']
        y_res = params['y_res']

        # linspace x and y dims
        x = np.linspace(0, x_range, x_res)
        y = np.linspace(0, y_range, y_res)
        # print('X = ', x)
        # print('Y = ', y)
        nodes = []
        nodes_dof = []
        load_cond_full = []
        for i in x:
            for j in y:

                # 1. Nodes
                nodes.append([i, j])

                # 2. Nodes DOF
                if i == min(x):
                    nodes_dof.append([0, 0])
                else:
                    nodes_dof.append([1, 1])

                # 3. Highlight all possible load locations with -1
                if i != min(x):
                    if j == min(y) or j == max(y) or i == max(x):
                        load_cond_full.append([0, -1])
                    else:
                        load_cond_full.append([0, 0])
                else:
                    load_cond_full.append([0, 0])

        load_conds = np.array(load_cond_full)
        # print(load_conds.shape)

        # Retrieve all indices in load_conds that have a -1
        load_conds_idx = np.where(load_conds[:, 1] == -1)
        # print('Load conds idx = ', load_conds_idx)

        load_conds_enum = []
        for idx in load_conds_idx[0]:
            load_cond_temp = np.zeros_like(load_conds)
            nodal_conds = deepcopy(all_node_load_conds)
            for i, cond in enumerate(nodal_conds):
                load_cond_temp[idx] = cond
                load_conds_enum.append(deepcopy(load_cond_temp))

        problem = {
            'nodes': nodes,
            'nodes_dof': nodes_dof,
            'member_radii': params['member_radii'],
            'youngs_modulus': params['youngs_modulus'],
        }
        problem_set = []
        for load_cond in load_conds_enum:
            p = deepcopy(problem)
            lc = deepcopy(load_cond)
            lc = list(lc)
            p['load_conds'] = [lc]
            if dropout > 0.0 and len(p['nodes']) > 4:
                # p = Cantilever.drop_nodes(p, dropout=dropout)
                # problem_set.append(p)
                p_set = AbstractProblem.drop_nodes_enum(p, dropout=dropout, ss=10)
                problem_set.extend(p_set)
            else:
                problem_set.append(p)


        if split is True:
            # Split into training and validation sets
            train_set = problem_set[:int(0.9*len(problem_set))]
            val_set = problem_set[int(0.9*len(problem_set)):]
            val_set_out = []
            return train_set, val_set, val_set_out
        else:
            return problem_set

    # ----------------------------------------------------------
    # Type 2
    #   Fixed Parameters
    #   - node mesh
    #   - member radii
    #   - young's modulus
    #   - fixed nodes (left side)
    #
    #   Enumerated Parameters
    #   - x node count
    #   - y node count
    #   - loaded node locations
    # ----------------------------------------------------------

    @staticmethod
    def type_2_enum(params, sample_size=64, dropout=0.0):
        x_range = params['x_range']
        y_range = params['y_range']
        x_res_range = params['x_res_range']
        y_res_range = params['y_res_range']
        enum_problems = []
        for x_res in range(x_res_range[0], x_res_range[1] + 1):
            for y_res in range(y_res_range[0], y_res_range[1] + 1):
                res_problems = Cantilever.type_1_enum({
                    'x_range': x_range,
                    'y_range': y_range,
                    'x_res': x_res,
                    'y_res': y_res,
                    'member_radii': params['radii'],
                    'youngs_modulus': params['y_modulus']
                }, dropout=dropout)
                if len(res_problems) > 0:
                    enum_problems.append(res_problems)
                    print(str(y_res) + 'x' + str(x_res), ':', len(res_problems))
                else:
                    print(str(y_res) + 'x' + str(x_res), ':', 0, '!!!')

        # In-bounds train and val problems
        train_problems = []
        val_problems = []
        for problem_enum in enum_problems:
            val_problem_idx = random.choice(range(len(problem_enum)))
            val_problems.append(problem_enum[val_problem_idx])
            problems_enum_train = [p for idx, p in enumerate(problem_enum) if idx != val_problem_idx]
            ss = min(len(problems_enum_train), sample_size)
            train_problems.extend(random.sample(problems_enum_train, ss))
            # train_problems.extend([p for idx, p in enumerate(problem_enum) if idx != val_problem_idx])

        # Out-of-bounds val problems
        temp_pset1 = Cantilever.type_1_enum({'x_range': x_range, 'y_range': y_range, 'x_res': x_res_range[1]+1, 'y_res': y_res_range[1], 'member_radii': params['radii'], 'youngs_modulus': params['y_modulus']})
        temp_pset2 = Cantilever.type_1_enum({'x_range': x_range, 'y_range': y_range, 'x_res': x_res_range[1]+1, 'y_res': y_res_range[0], 'member_radii': params['radii'], 'youngs_modulus': params['y_modulus']})
        temp_pset3 = Cantilever.type_1_enum({'x_range': x_range, 'y_range': y_range, 'x_res': x_res_range[1], 'y_res': y_res_range[1]+1, 'member_radii': params['radii'], 'youngs_modulus': params['y_modulus']})
        temp_pset4 = Cantilever.type_1_enum({'x_range': x_range, 'y_range': y_range, 'x_res': x_res_range[0], 'y_res': y_res_range[1]+1, 'member_radii': params['radii'], 'youngs_modulus': params['y_modulus']})
        temp_pset5 = Cantilever.type_1_enum({'x_range': x_range, 'y_range': y_range, 'x_res': x_res_range[1]+1, 'y_res': y_res_range[1]+1, 'member_radii': params['radii'], 'youngs_modulus': params['y_modulus']})
        val_problems_out = [random.choice(temp_pset1), random.choice(temp_pset2), random.choice(temp_pset3), random.choice(temp_pset4), random.choice(temp_pset5)]

        return train_problems, val_problems, val_problems_out





def get_ppath():
    ppath = os.path.join(config.plots_dir, 'problems', 'cantilever')
    if not os.path.exists(ppath):
        os.makedirs(ppath)
    # find how many directories are in the ppath
    dir_count = len([name for name in os.listdir(ppath) if os.path.isdir(os.path.join(ppath, name))])
    dir_path = os.path.join(ppath, 'problem_space_' + str(dir_count))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path



if __name__ == '__main__':
    from combench.models import truss
    t_problems, v_problems, v_problems_out = get_problems()
    # exit(0)
    b_dir = get_ppath()

    for idx, vp_out in enumerate(v_problems_out):
        truss.set_norms(vp_out)
        design_rep = [1 for _ in range(truss.rep.get_num_bits(vp_out))]
        f_name = f'val_out_{idx}.png'
        truss.rep.viz(vp_out, design_rep, f_name=f_name, base_dir=b_dir)

    for idx, vp in enumerate(v_problems):
        truss.set_norms(vp)
        design_rep = [1 for _ in range(truss.rep.get_num_bits(vp))]
        f_name = f'val_{idx}.png'
        truss.rep.viz(vp, design_rep, f_name=f_name, base_dir=b_dir)










