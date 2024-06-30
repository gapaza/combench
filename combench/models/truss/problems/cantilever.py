from combench.models.truss.problems.abstract_problem import AbstractProblem
import numpy as np
from copy import deepcopy
import random
import math

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




class Cantilever(AbstractProblem):


    def __init__(self):
        super().__init__()

    @staticmethod
    def random_problem(params):
        pass

    @staticmethod
    def enumerate_res(params, sample_size=64, seed=0, dropout=0.0):
        x_range = params['x_range']
        y_range = params['y_range']
        x_res_range = params['x_res_range']
        y_res_range = params['y_res_range']
        enum_problems = []
        for x_res in range(x_res_range[0], x_res_range[1] + 1):
            for y_res in range(y_res_range[0], y_res_range[1] + 1):
                res_problems = Cantilever.enumerate({
                    'x_range': x_range,
                    'y_range': y_range,
                    'x_res': x_res,
                    'y_res': y_res,
                    'member_radii': params['radii'],
                    'youngs_modulus': params['y_modulus']
                }, dropout=dropout)
                enum_problems.append(res_problems)
                print(str(y_res) + 'x' +  str(x_res), ':', len(res_problems))

        # In-bounds train and val problems
        train_problems = []
        val_problems = []
        random.seed(seed)
        for problem_enum in enum_problems:
            val_problem_idx = random.choice(range(len(problem_enum)))
            val_problems.append(problem_enum[val_problem_idx])
            problems_enum_train = [p for idx, p in enumerate(problem_enum) if idx != val_problem_idx]
            ss = min(len(problems_enum_train), sample_size)
            train_problems.extend(random.sample(problems_enum_train, ss))
            # train_problems.extend([p for idx, p in enumerate(problem_enum) if idx != val_problem_idx])

        # Out-of-bounds val problems
        temp_pset1 = Cantilever.enumerate({'x_range': x_range, 'y_range': y_range, 'x_res': x_res_range[1]+1, 'y_res': y_res_range[1], 'member_radii': params['radii'], 'youngs_modulus': params['y_modulus']})
        temp_pset2 = Cantilever.enumerate({'x_range': x_range, 'y_range': y_range, 'x_res': x_res_range[1]+1, 'y_res': y_res_range[0], 'member_radii': params['radii'], 'youngs_modulus': params['y_modulus']})
        temp_pset3 = Cantilever.enumerate({'x_range': x_range, 'y_range': y_range, 'x_res': x_res_range[1], 'y_res': y_res_range[1]+1, 'member_radii': params['radii'], 'youngs_modulus': params['y_modulus']})
        temp_pset4 = Cantilever.enumerate({'x_range': x_range, 'y_range': y_range, 'x_res': x_res_range[0], 'y_res': y_res_range[1]+1, 'member_radii': params['radii'], 'youngs_modulus': params['y_modulus']})
        temp_pset5 = Cantilever.enumerate({'x_range': x_range, 'y_range': y_range, 'x_res': x_res_range[1]+1, 'y_res': y_res_range[1]+1, 'member_radii': params['radii'], 'youngs_modulus': params['y_modulus']})
        val_problems_out = [random.choice(temp_pset1), random.choice(temp_pset2), random.choice(temp_pset3), random.choice(temp_pset4), random.choice(temp_pset5)]

        return train_problems, val_problems, val_problems_out




    # This generates cantilever problems where the mesh and fixed nodes are static, and the load conditions are random
    @staticmethod
    def enumerate(params, dropout=0.0):
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
                

        from combench.models.truss.representation import get_edge_nodes, get_load_nodes, get_all_fixed_nodes
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

            if dropout > 0.0:
                # print('Load cond = ', load_cond)
                n_nodes = len(p['nodes'])
                edge_nodes = get_edge_nodes(p)
                load_nodes = get_load_nodes(lc)
                fixed_nodes = get_all_fixed_nodes(p)
                other_nodes = [x for x in range(n_nodes) if x not in (edge_nodes + load_nodes + fixed_nodes)]

                droppable_nodes = []
                droppable_nodes.extend(other_nodes)
                if len(fixed_nodes) > 2:
                    droppable_nodes.extend(fixed_nodes[:-2])
                non_load_edge_nodes = [x for x in edge_nodes if x not in (load_nodes + fixed_nodes)]
                droppable_nodes.extend(non_load_edge_nodes)

                droppable_nodes.sort(reverse=True)
                for drop_node in droppable_nodes:
                    if random.random() < dropout:
                        del p['nodes'][drop_node]
                        del p['nodes_dof'][drop_node]
                        del lc[drop_node]

            p['load_conds'] = [lc]
            problem_set.append(p)

        return problem_set

    @staticmethod
    def get_problem(load_conds, x_range=4, y_range=2, x_res=4, y_res=2, radii=0.1, y_modulus=1e9):

        # linspace x and y dims
        x = np.linspace(0, x_range, x_res)
        y = np.linspace(0, y_range, y_res)
        # print('X = ', x)
        # print('Y = ', y)
        nodes = []
        nodes_dof = []
        for i in x:
            for j in y:
                nodes.append([i, j])
                if i == min(x):
                    nodes_dof.append([0, 0])
                else:
                    nodes_dof.append([1, 1])

        # Load cases
        if load_conds is None:
            load_conds = []
            for idx, n in enumerate(nodes):
                if n[0] == max(x):
                    if n[1] == min(y) or n[1] == max(y):
                        load_conds.append([0, -1])
                    else:
                        load_conds.append([0, -1])
                else:
                    load_conds.append([0, 0])
            all_load_conds = [load_conds]
        else:
            all_load_conds = load_conds

        problem = {
            'nodes': nodes,
            'nodes_dof': nodes_dof,
            'load_conds': all_load_conds,
            'member_radii': radii,
            'youngs_modulus': y_modulus,
        }

        return problem



def get_problems(seed=0):
    from combench.models import truss
    train_problems, val_problems, val_problems_out = truss.problems.Cantilever.enumerate_res({
        'x_range': 4,
        'y_range': 4,
        'x_res_range': [2, 4],
        'y_res_range': [2, 4],
        'radii': 0.2,
        'y_modulus': 210e9
    }, dropout=0.1)

    n_cnt = []
    for t in train_problems:
        if len(t['nodes']) not in n_cnt:
            n_cnt.append(len(t['nodes']))
    print('Train problems = ', n_cnt)


    # print('Train problems = ', len(train_problems))
    # print('Val problems = ', len(val_problems))
    # shuffle train problems
    random.shuffle(train_problems)
    for idx, problem in enumerate(train_problems):
        design_rep = [1 for x in range(truss.rep.get_num_bits(problem))]
        truss.rep.viz(problem, design_rep, f_name='train_' + str(idx) + '.png')
        if idx > 5:
            break
    # for idx, problem in enumerate(val_problems):
    #     design_rep = [1 for x in range(truss.rep.get_num_bits(problem))]
    #     truss.rep.viz(problem, design_rep, f_name='val_' + str(idx) + '.png')
    # for idx, problem in enumerate(val_problems_out):
    #     design_rep = [1 for x in range(truss.rep.get_num_bits(problem))]
    #     truss.rep.viz(problem, design_rep, f_name='val_out_' + str(idx) + '.png')










if __name__ == '__main__':
    from combench.models import truss
    problem_set = get_problems()






