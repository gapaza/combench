from combench.models.truss.problems.abstract_problem import AbstractProblem
import numpy as np
from copy import deepcopy
import random
from combench.models import truss
from itertools import combinations

node_loads_enum = [
    [-1, 0],
    # [0, -1],
    # [1, 0],
    # [0, 1],
    # [1, 1],
    [-1, -1],
    # [1, -1],
    [-1, 1]
]

RM_NODES = False


class TrussType1(AbstractProblem):

    def __init__(self):
        super().__init__()

    @staticmethod
    def enumerate(params, p_type='type1'):
        if p_type == 'type1':
            return TrussType1.type_1_enum(params)
        elif p_type == 'type2':
            return TrussType1.type_2_enum(params)
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
    #
    #   Enumerated Parameters
    #   - fixed node locations (2 fixed)
    #   - loaded node locations
    # ----------------------------------------------------------

    @staticmethod
    def type_1_enum(params):
        x_range = params['x_range']
        y_range = params['y_range']
        x_res = params['x_res']
        y_res = params['y_res']

        # 1. Get node mesh and edge nodes
        nodes = AbstractProblem.get_mesh(x_range, x_res, y_range, y_res)
        nodes_idx = [x for x in range(len(nodes))]
        # middle_node_idx = int(len(nodes) // 2)
        nodes_dof = [[1, 1] for x in range(len(nodes))]
        nodes_loads = [[0, 0] for x in range(len(nodes))]
        base_problem = {
            'nodes': nodes,
            'nodes_dof': nodes_dof,
            'load_conds': [nodes_loads],
            'member_radii': params['radii'],
            'youngs_modulus': params['y_modulus']
        }
        edge_nodes_idx = truss.rep.get_edge_nodes(base_problem)

        # 2. Find combinations of edge nodes
        edge_nodes_idx_combs = list(combinations(edge_nodes_idx, 2))

        # 3. Iterate over combinations and create problems
        enum_problems = []
        for edge_node_comb in edge_nodes_idx_combs:
            fixed_1 = edge_node_comb[0]
            fixed_2 = edge_node_comb[1]

            # Iterate over edge nodes that are not in the combination
            for node_idx in edge_nodes_idx:
                if node_idx not in edge_node_comb:
                    loaded_node_idx = node_idx

                    # Iterate over possible node loads
                    for node_load in node_loads_enum:
                        problem = deepcopy(base_problem)
                        problem['nodes_dof'][fixed_1] = [0, 0]
                        problem['nodes_dof'][fixed_2] = [0, 0]
                        problem['load_conds'][0][loaded_node_idx] = deepcopy(node_load)
                        enum_problems.append(problem)

                        if RM_NODES:
                            rm_node = random.choice(edge_nodes_idx)
                            while rm_node in [fixed_1, fixed_2, loaded_node_idx]:
                                rm_node = random.choice(edge_nodes_idx)
                            # Delete this index from the nodes, nodes dof, and load conds
                            del problem['nodes'][rm_node]
                            del problem['nodes_dof'][rm_node]
                            del problem['load_conds'][0][rm_node]

        return enum_problems

    # ----------------------------------------------------------
    # Type 2
    #   Fixed Parameters
    #   - node mesh
    #   - member radii
    #   - young's modulus
    #
    #   Enumerated Parameters
    #   - x node count
    #   - y node count
    #   - fixed node locations (2 fixed)
    #   - loaded node locations
    # ----------------------------------------------------------

    @staticmethod
    def type_2_enum(params, sample_size=64):
        x_range = params['x_range']
        y_range = params['y_range']
        x_res_range = params['x_res_range']
        y_res_range = params['y_res_range']
        enum_problems = []
        for x_res in range(x_res_range[0], x_res_range[1] + 1):
            for y_res in range(y_res_range[0], y_res_range[1] + 1):
                res_problems = TrussType1.type_1_enum({
                    'x_range': x_range,
                    'y_range': y_range,
                    'x_res': x_res,
                    'y_res': y_res,
                    'radii': params['radii'],
                    'y_modulus': params['y_modulus']
                })
                enum_problems.append(res_problems)
                # print('Problem set:', len(res_problems), 'for res:', x_res, y_res)
        all_problems = []
        for problem_enum in enum_problems:
            ss = min(sample_size, len(problem_enum))
            all_problems.extend(random.sample(problem_enum, ss))

        # print('Problem set:', len(all_problems))
        return all_problems



def res_full():
    problem_set = TrussType1.type_2_enum({
        'x_range': 4,
        'y_range': 4,
        'x_res_range': [2, 4],
        'y_res_range': [2, 4],
        'radii': 0.2,
        'y_modulus': 210e9
    })
    random.seed(22)
    problem_set = random.sample(problem_set, 256)
    design_rep = [1 for x in range(truss.rep.get_num_bits(problem_set[0]))]
    truss.rep.viz(problem_set[0], design_rep, f'problems/{TrussType1.__name__}RES.png')

    exit(0)


if __name__ == '__main__':
    # res_full()

    N = 3
    problem_set = TrussType1.enumerate({
        'x_range': N,
        'y_range': N,
        'x_res': N,
        'y_res': N,
        'radii': 0.2,
        'y_modulus': 210e9
    })
    print('Problem set:', len(problem_set))

    problem = problem_set[0]
    design = [1 for x in range(truss.rep.get_num_bits(problem))]
    truss.rep.viz(problem, design, f_name='truss_type1.png')

