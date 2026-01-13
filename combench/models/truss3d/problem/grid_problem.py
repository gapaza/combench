import math
from copy import deepcopy

import numpy as np

from combench.models.truss3d.problem.abstract_problem import AbstractProblem
from combench.models.truss3d import representation as rep


class GridProblem(AbstractProblem):


    def __init__(self, x_range, x_res, y_range, y_res, z_range, z_res):
        super().__init__()
        self.x_range = x_range
        self.x_res = x_res
        self.y_range = y_range
        self.y_res = y_res
        self.z_range = z_range
        self.z_res = z_res


        self.grid_nodes = self.get_mesh(
            x_range,
            x_res,
            y_range,
            y_res,
            z_range,
            z_res
        )




    def generate(self, fixed_nodes, load_dict, radii=0.1, y_modulus=1.8162e6):
        """Generates a grid-based truss problem with boundary conditions.

        Args:
            fixed_nodes (np.array): Indices of fixed nodes.
            load_dict dict: Maps node indices to load vectors.

        Returns:
            problem (dict): A dictionary encoding the truss problem.
        """

        nodes = deepcopy(self.grid_nodes) # List of tuples

        nodes_dof = np.ones_like(np.array(nodes))
        nodes_dof[fixed_nodes] = [0, 0, 0]

        load_conds = np.zeros_like(np.array(nodes))
        for key, vec in load_dict.items():
            load_conds[key, :] = vec

        return {
            'nodes': nodes,
            'nodes_dof': nodes_dof.tolist(),
            'load_conds': [load_conds.tolist()],

            'member_radii': radii,
            'youngs_modulus': y_modulus,
        }






























# -------------------------------------------------------
# Testing
# -------------------------------------------------------
from combench.models.truss3d.stiffness.truss_model import eval_stiffness
from combench.models.truss3d.vol.naive import eval_volfrac


if __name__ == '__main__':


    range_x = 10.0
    res_x = 3
    range_y = 10.0
    res_y = 3
    range_z = 10.0
    res_z = 3

    problem_gen = GridProblem(range_x, res_x, range_y, res_y, range_z, res_z)


    fixed_nodes = np.array([0, 17, 21])
    load_conds = {
        9: [0, 0, -1000.0],
    }
    problem = problem_gen.generate(fixed_nodes, load_conds)


    n_bits = rep.get_num_bits(problem)
    print('Num bits:', n_bits)
    design = [1 for _ in range(n_bits)]

    compliance = eval_stiffness(problem, design, problem['load_conds'][0])
    volfrac = eval_volfrac(problem, design)
    print('Compliance:', compliance)
    print('Volfrac:', volfrac)

    rep.viz3d(problem, design, 'grid_problem.png')