import numpy as np
import math
from copy import deepcopy
import scipy.linalg
import pyamg

from combench.models.truss3d import representation as rep
from combench.models.truss3d.stiffness.formK import formK as getK


"""
----- Design Representations -----

 - Bit List: [0, 1, 1, 1, 0, 0, ..., 1]
 - Bit Str: '011100...1'
 - List of node index pairs: [
          [0, 1],
          [0, 2]
    ]
 - List of node coordinates: [
        [ [0, 1, 1], [2, 1, 1] ] ,
        [ [0, 1, 1], [3, 1. 1] ]
    ]


----- Problem Data Structure -----

 - Nodes must always be ordered by x-coordinate, then y-coordinate
    problem = {
        'nodes': [  # Node coordinate system is in meters
            (0, 0, 0), (0, 1, 1),
            (1, 0, 0), (1, 1, 0),
            (2, 0, 0), (2, 1, 0),
            (3, 0, 0), (3, 1, 0),
        ],
        'nodes_dof': [  # Encodes which degrees of freedom are fixed for each node
            (0, 0, 0), (1, 1, 1),
            (0, 0, 0), (1, 1, 1),
            (0, 0, 0), (1, 1, 1),
            (0, 0, 0), (1, 1, 1),
        ],
        'load_conds': [ # Encodes the loads applied to each node in each direction (newtons)
            [  # Multiple load conditions can be specified
                (0, 0, 0), (0, 0, 0),
                (0, 0, 0), (0, 0, 0),
                (0, 0, 0), (0, 0, 0),
                (0, 0, 0), (1, 1, 1),
            ]
        ],
        'member_radii': 0.1,         # Radii is in meters
        'youngs_modulus': 1.8162e6,  # Material youngs modulus is in pascales (N/m^2)
}
"""


def eval_stiffness(problem, design_rep, load_cond, iter_solve=False, stress_calc=False):
    extra_info = {}

    # ------------------------------------------
    # Design Validation
    # ------------------------------------------

    # 1. Validate design has at least one member
    bit_list, bit_str, node_idx_pairs, node_coords = rep.convert(problem, design_rep)
    if sum(bit_list) == 0:
        extra_info['Error'] = 'No members in design'
        return 1e9, extra_info

    load_nodes = rep.get_load_nodes(load_cond)
    used_nodes_idx = rep.get_used_nodes(problem, bit_list)
    fixed_nodes = rep.get_all_fixed_nodes(problem)

    # 2. Validate all nodes with loads are used
    for ln in load_nodes:
        if ln not in used_nodes_idx:
            extra_info['Error'] = 'Load node {} not used'.format(ln)
            return 1e9, extra_info

    # 3. Validate at least one fixed node is used
    fixed_node_used = False
    for fn in fixed_nodes:
        if fn in used_nodes_idx:
            fixed_node_used = True
            break
    if not fixed_node_used:
        extra_info['Error'] = 'No fixed nodes used'
        return 0, extra_info

    # ------------------------------------------
    # 2. Optimization: Reduce Problem Size
    # ------------------------------------------

    # Identify Active Elements
    # Filter connectivity to keep only enabled members
    active_pairs_global = node_idx_pairs

    # Identify Active Nodes
    # We use a set to find the unique nodes involved in the active structure
    active_nodes_set = set()
    for n1, n2 in active_pairs_global:
        active_nodes_set.add(n1)
        active_nodes_set.add(n2)

    sorted_active_nodes = sorted(list(active_nodes_set))
    num_active_nodes = len(sorted_active_nodes)

    # Create Mapping: Global Node Index -> Local Reduced Index
    # This allows us to build a matrix of size (3*N_active) instead of (3*N_total)
    global_to_local = {global_id: local_id for local_id, global_id in enumerate(sorted_active_nodes)}

    # ------------------------------------------
    # 3. Assemble Reduced System Arrays
    # ------------------------------------------

    # Reduced Nodal Coordinates (NC)
    NC_reduced = np.array([problem['nodes'][i] for i in sorted_active_nodes])

    # Reduced Connectivity Array (CA) using local indices
    CA_reduced = np.array([[global_to_local[n1], global_to_local[n2]] for n1, n2 in active_pairs_global])

    # Areas (Avar)
    # Calculate area from radii (A = pi * r^2)
    area = np.pi * (problem['member_radii'] ** 2)
    Avar = np.full(len(active_pairs_global), area)

    # Material Properties
    E = problem['youngs_modulus']

    # ------------------------------------------
    # 4. Build Global Stiffness Matrix (K)
    # ------------------------------------------

    # Call your existing helper method with the reduced arrays
    K_reduced = getK(NC_reduced, CA_reduced, Avar, E)

    # ------------------------------------------
    # 5. Apply Boundary Conditions & Loads
    # ------------------------------------------

    free_dofs = []
    F_reduced = np.zeros(3 * num_active_nodes)

    # Iterate through active nodes to populate Force vector and identify Free DOFs
    for local_i, global_i in enumerate(sorted_active_nodes):

        # Get constraints and loads from the original problem definition
        # Assumes: 1 = Free (Active), 0 = Fixed (Restrained)
        dof_flags = problem['nodes_dof'][global_i]
        applied_load = load_cond[global_i]

        for dim in range(3):  # x, y, z
            dof_index = 3 * local_i + dim

            # Populate Force Vector
            F_reduced[dof_index] = applied_load[dim]

            # Identify Free Degrees of Freedom
            if dof_flags[dim] == 1:
                free_dofs.append(dof_index)

    if not free_dofs:
        extra_info['Error'] = 'Structure is fully constrained (no DOFs)'
        return 0.0, extra_info

    # ------------------------------------------
    # 6. Solve Linear System (K * u = F)
    # ------------------------------------------

    try:
        # Partition the system to solve only for Free DOFs
        # K_ff: Stiffness matrix rows/cols corresponding to free DOFs
        # F_f:  Force vector rows corresponding to free DOFs

        # np.ix_ is used to construct the index mesh for slicing
        K_ff = K_reduced[np.ix_(free_dofs, free_dofs)]
        F_f = F_reduced[free_dofs]

        # Solve for displacements
        if iter_solve is False:
            # assume_a='sym' tells scipy the matrix is symmetric, which is faster
            u_f = scipy.linalg.solve(K_ff, F_f, assume_a='sym')
        else:
            ml = pyamg.ruge_stuben_solver(K_ff)
            u_f = ml.solve(F_f, tol=1e-8)

        # ------------------------------------------
        # 7. Calculate Compliance
        # ------------------------------------------

        # Compliance = Force_transpose * Displacement
        # This is the standard scalar measure for "stiffness" in optimization
        compliance = np.dot(F_f, u_f)

        # ------------------------------------------
        # 8. Stress Calculation (Optional)
        # ------------------------------------------

        if stress_calc is True:

            # 1. Reconstruct the full displacement vector for all active nodes
            u_reduced = np.zeros(3 * num_active_nodes)
            u_reduced[free_dofs] = u_f

            member_stresses = []
            member_buckling_ratios = []  # Actual Stress / Buckling Stress

            # 2. Iterate through each active member
            for i, (g_idx1, g_idx2) in enumerate(active_pairs_global):
                # Get local indices in the reduced system
                l_idx1, l_idx2 = global_to_local[g_idx1], global_to_local[g_idx2]

                # Get nodal coordinates
                p1 = NC_reduced[l_idx1]
                p2 = NC_reduced[l_idx2]

                # Calculate geometry
                vec = p2 - p1
                L = np.linalg.norm(vec)
                unit_vec = vec / L

                # Extract displacements for these two nodes (3 DOFs each)
                u1 = u_reduced[3 * l_idx1: 3 * l_idx1 + 3]
                u2 = u_reduced[3 * l_idx2: 3 * l_idx2 + 3]

                # Calculate relative displacement along the member axis
                delta_L = np.dot((u2 - u1), unit_vec)

                # Calculate Normal Stress (Positive = Tension, Negative = Compression)
                # Sigma = E * epsilon = E * (delta_L / L)
                stress = E * (delta_L / L)
                member_stresses.append(stress)

                # 3. Buckling Evaluation (Only relevant for compression)
                r = problem['member_radii']
                # Critical Buckling Stress (Euler)
                sigma_cr = (np.pi ** 2 * E * (r ** 2)) / (4 * L ** 2)

                if stress < 0:  # Member is in compression
                    buckling_ratio = abs(stress) / sigma_cr
                else:
                    buckling_ratio = 0.0

                member_buckling_ratios.append(buckling_ratio)

            extra_info['member_stresses'] = np.array(member_stresses)
            extra_info['buckling_ratios'] = np.array(member_buckling_ratios)

            # Check for failure (if any member exceeds buckling limit)
            if any(r > 1.0 for r in member_buckling_ratios):
                extra_info['Status'] = 'Buckling Failure'

        return compliance, extra_info

    except np.linalg.LinAlgError:
        # This catches singular matrices (e.g., mechanisms, floating nodes)
        extra_info['Error'] = 'Singular Matrix (Unstable Structure)'
        return 1e9, extra_info





if __name__ == '__main__':

    problem = {
            'nodes': [  # Node coordinate system is in meters
                (0, 0, 0), (0, 1, 1),
                (1, 0, 0), (1, 1, 0),
                (2, 0, 0), (2, 1, 0),
                (3, 0, 0), (3, 1, 0),
            ],
            'nodes_dof': [  # Encodes which degrees of freedom are fixed for each node
                (0, 0, 0), (0, 0, 0),
                (1, 1, 1), (0, 0, 0),
                (1, 1, 1), (1, 1, 1),
                (1, 1, 1), (1, 1, 1),
            ],
            'load_conds': [ # Encodes the loads applied to each node in each direction (newtons)
                [  # Multiple load conditions can be specified
                    (0, 0, 0), (0, 0, 0),
                    (0, 0, 0), (0, 0, 0),
                    (0, 0, 0), (0, 0, 0),
                    (0, 0, 0), (100, 0, 0),
                ]
            ],
            'member_radii': 0.1,         # Radii is in meters
            'youngs_modulus': 1.8162e6,  # Material youngs modulus is in pascales (N/m^2)
    }

    n_bits = rep.get_num_bits(problem)
    d1_sample = [1 for x in range(n_bits)]
    # d1_sample = rep.random_sample_1(problem)

    compliance, info = eval_stiffness(problem, d1_sample, problem['load_conds'][0], iter_solve=False)
    print('Compliance:', compliance)
    print('Info:', info)

     # Visualize
    rep.viz3d(problem, d1_sample, 'd1_sample.png')









