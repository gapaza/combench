from combench.models.truss.problems.abstract_problem import AbstractProblem
import numpy as np
from copy import deepcopy
import random



class Cantilever(AbstractProblem):


    def __init__(self):
        super().__init__()

    @staticmethod
    def random_problem(params):
        pass

    @staticmethod
    def type_1_enum(params):
        type_1_problem = Cantilever.type_1(params)
        problems = []
        for load_conn in type_1_problem['load_conds_enum']:
            problem = {
                'nodes': type_1_problem['nodes'],
                'nodes_dof': type_1_problem['nodes_dof'],
                'load_conds': [load_conn],
                'member_radii': type_1_problem['member_radii'],
                'youngs_modulus': type_1_problem['youngs_modulus'],
            }
            problems.append(problem)

        # shuffle problems to avoid bias
        # random.seed(42)
        # random.shuffle(problems)

        return problems





    # This generates cantileve problems where the mesh and fixed nodes are static, and the load conditions are random
    @staticmethod
    def type_1(params):
        x_range = params['x_range']
        y_range = params['y_range']
        x_res = params['x_res']
        y_res = params['y_res']

        # linspace x and y dims
        x = np.linspace(0, x_range, x_res)
        y = np.linspace(0, y_range, y_res)
        print('X = ', x)
        print('Y = ', y)
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

                # 3. Load conditions
                if i != min(x):
                    if j == min(y) or j == max(y) or i == max(x):
                        load_cond_full.append([0, -1])
                    else:
                        load_cond_full.append([0, 0])
                else:
                    load_cond_full.append([0, 0])

        load_conds = np.array(load_cond_full)
        print(load_conds.shape)

        # Retrieve all indices in load_conds that have a -1
        load_conds_idx = np.where(load_conds[:, 1] == -1)
        # print('Load conds idx = ', load_conds_idx)

        load_conds_enum = []
        for idx in load_conds_idx[0]:
            load_cond_temp = np.zeros_like(load_conds)
            nodal_conds = [
                [-1, 0],
                [0, -1],
                [1, 0],
                [0, 1],
                [1, 1],
                [-1, -1]
            ]
            for i, cond in enumerate(nodal_conds):
                load_cond_temp[idx] = cond
                load_conds_enum.append(deepcopy(load_cond_temp))

                # print('Load cond = ', load_cond_temp)
        print('Total load conds = ', len(load_conds_enum))



        all_load_conds = [
            # random.choice(load_conds_enum),
            # random.choice(load_conds_enum),
            # random.choice(load_conds_enum),
            load_conds_enum[0],  # load_conds_enum[1], load_conds_enum[2], load_conds_enum[-1]
        ]

        # all_load_conds = [load_cond_full]

        problem = {
            'nodes': nodes,
            'nodes_dof': nodes_dof,
            'load_conds': all_load_conds,
            'load_conds_enum': load_conds_enum,
            'member_radii': params['radii'],
            'youngs_modulus': params['y_modulus'],
        }

        return problem

    @staticmethod
    def get_problem(load_conds, x_range=4, y_range=2, x_res=4, y_res=2, radii=0.1, y_modulus=1e9):

        # linspace x and y dims
        x = np.linspace(0, x_range, x_res)
        y = np.linspace(0, y_range, y_res)
        print('X = ', x)
        print('Y = ', y)
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







if __name__ == '__main__':
    problem = Cantilever.get_problem(
        None,
        x_range=4,
        y_range=3,
        x_res=4,
        y_res=3,
        radii=0.2,
        y_modulus=210e9
    )
    from combench.models.truss import rep
    # design_rep = rep.grid_design_sample(problem)
    # design_rep = [1 for x in range(rep.get_num_bits(problem))]

    # design_str = '101000100000011000001001010000001101101001001011010001000001001100'
    # design_rep = [int(x) for x in design_str]

    design_rep = [int(1) for x in range(rep.get_num_bits(problem))]
    design_rep = rep.remove_overlapping_members(problem, design_rep)

    # for idx, node in enumerate(problem['nodes']):
    #     print(f'Node {idx}:', rep.get_node_connections(problem, design_rep, idx))


    rep.viz(problem, design_rep, f'problems/{Cantilever.__name__}.png')





