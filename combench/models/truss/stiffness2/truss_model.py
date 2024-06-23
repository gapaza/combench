import numpy as np
from combench.models.truss.stiffness2.formK import fullK
import config
from combench.models.truss import representation as rep
import combench.models.truss as truss
from copy import deepcopy

""" Problem Input
problem = {
    'nodes': [  # Node coordinate system is in meters
        (0, 0), (0, 1), 
        (1, 0), (1, 1), (1, 2), (1, 3),
        (2, 0), (2, 1), (2, 2), (2, 3),
        (3, 0), (3, 1), (3, 2), (3, 3),
    ],
    'nodes_dof': [  # Encodes which degrees of freedom are fixed for each node
        (0, 0), (1, 1), (1, 1), (1, 1),
        (0, 0), (1, 1), (1, 1), (1, 1),
        (0, 0), (1, 1), (1, 1), (1, 1),
        (0, 0), (1, 1), (1, 1), (1, 1),
    ],
    'load_conds': [ # Encodes the loads applied to each node in each direction (newtons)
        [  # Multiple load conditions can be specified
            (0, 0), (0, 0),
            (1, 1), (1, 1),
            (1, 1), (1, 1),
        ]
    ],
    'member_radii': 0.1,         # Radii is in meters
    'youngs_modulus': 1.8162e6,  # Material youngs modulus is in pascales (N/m^2)
}
"""

DEBUG = False

def eval_load_cond(problem, design_rep, load_conds):
    if DEBUG:
        print('\n\n----------- Evaluating Load Condition -----------')
    if problem is None:
        problem = truss.problem
    for expected_key in ['nodes', 'nodes_dof', 'member_radii', 'youngs_modulus']:
        if expected_key not in problem:
            raise ValueError(f"Expected key '{expected_key}' not found in problem")

    # ------------------------------------------
    # Design Validation
    # ------------------------------------------

    # 1. Validate design has at least one member
    bit_list, bit_str, node_idx_pairs, node_coords = rep.convert(problem, design_rep)
    if sum(bit_list) == 0:
        return 0
    load_nodes = rep.get_load_nodes(load_conds)
    used_nodes = rep.get_used_nodes(problem, bit_list)
    fixed_nodes = rep.get_all_fixed_nodes(problem)
    if DEBUG:
        print('Used Nodes:', used_nodes)
        print('Fixed Nodes:', fixed_nodes)
        print('Load Nodes:', load_nodes)

    # 2. Validate node constraints
    for node_idx in used_nodes:
        node_connections = rep.get_node_connections(problem, design_rep, node_idx)
        if len(node_connections) <= 1:
            return 0

    # 3. Validate all nodes with loads are used
    for ln in load_nodes:  # All load nodes with loads
        if ln not in used_nodes:
            return 0

    # 4. Validate at least one fixed node is used
    fixed_node_used = False
    for fn in fixed_nodes:
        if fn in used_nodes:
            fixed_node_used = True
            break
    if not fixed_node_used:
        return 0

    # ------------------------------------------
    # Stiffness Calculation
    # ------------------------------------------
    all_nodes = problem['nodes']

    # 1. Filter out unused nodes
    node_mapping = {}  # Maps index of problem nodes to index of used nodes
    nodes = []
    nodes_dof = []  # problem['nodes_dof']
    nodes_loads = []
    for idx, node in enumerate(all_nodes):
        if idx in used_nodes:
            nodes.append(node)
            nodes_dof.append(problem['nodes_dof'][idx])
            nodes_loads.append(load_conds[idx])
            node_mapping[idx] = len(nodes) - 1
    node_mapping_rev = {v: k for k, v in node_mapping.items()}

    # 2. Get design node_idx pairs for used nodes indices
    node_idx_pairs_mapped = []
    for idx_pair in node_idx_pairs:
        node_idx_pairs_mapped.append((node_mapping[idx_pair[0]], node_mapping[idx_pair[1]]))

    # 3. Find the cross sectional area of each member
    m_rad = problem['member_radii']
    m_areas = np.array([np.pi * m_rad ** 2 for _ in range(len(node_idx_pairs_mapped))])

    # 4. Add indices to nodes and members (elements)
    new_coords = []
    for idx, nc in enumerate(nodes):
        new_coords.append([idx + 1, nc[0], nc[1]])
    nodes = np.array(new_coords)

    new_conns = []
    for idx, nc in enumerate(node_idx_pairs_mapped):
        new_conns.append([idx + 1, nc[0], nc[1]])
    elements = np.array(new_conns)

    # 5. Calculate the global stiffness_old matrix (check if singular)
    y_mod = problem['youngs_modulus']
    K = fullK(nodes, elements, m_areas, y_mod)  # Global stiffness_old matrix
    K_det = np.linalg.det(K)
    if DEBUG:
        print('Global stiffness matrix determinant:', K_det, np.isclose(K_det, 0))
    if np.isclose(K_det, 0):
        return 0

    # 6. Create a mask to filter out fixed nodes (working with used nodes only)
    # - 0: fixed, 1: free
    mask_fixed_nodes = []
    for node_dof in nodes_dof:
        mask_fixed_nodes.extend(list(node_dof))
    num_free = sum(mask_fixed_nodes)
    mask_fixed_nodes = np.array(mask_fixed_nodes).astype(bool)

    # 7. Create mask to filter out x and y nodes
    mask_y = (np.arange(1, len(mask_fixed_nodes) + 1) % 2).astype(bool)  # [1, 0, 1, 0, 1, 0, ...]
    mask_x = np.logical_not(mask_y).astype(bool)                         # [0, 1, 0, 1, 0, 1, ...]

    # 8. Flatten node loads (working with used nodes only)
    flat_loads = []
    flat_loads_used = []
    for node_load in nodes_loads:
        flat_loads.extend(list(node_load))
        flat_loads_used.extend(list(node_load))

    # 9. Get loads applied to free nodes only
    flat_loads = np.array(flat_loads)
    flat_loads = flat_loads[mask_fixed_nodes]

    # 10. Create force vector on free nodes
    Ff = np.ones(shape=(num_free,))
    Ff = Ff * flat_loads

    # 11. Extract free nodes from global stiffness_old matrix
    Kfree = K[mask_fixed_nodes]
    Kfree = Kfree[:, mask_fixed_nodes]
    if DEBUG:
        print('Kfree:', Kfree.shape, Kfree)

    # 12. Solve for displacements of free nodes
    try:
        uf = np.linalg.solve(Kfree, Ff)
    except Exception as e:
        if DEBUG:
            print('Linear alg error:', e)
        return 0

    # 13. Construct full displacement vector with fixed nodes (working with used nodes only)
    u = np.zeros(2 * len(nodes))
    u[mask_fixed_nodes] = uf  # Set free node displacements

    # 14. Calculate all nodal forces with full displacement vector
    try:
        F = K @ u
    except Exception as e:
        if DEBUG:
            print('Linear alg error:', e)
        return 0

    # ------------------------------------------
    # DEBUG SOLVER
    # ------------------------------------------

    if DEBUG:
        u_node_num = 0  # in used nodes frame
        for x in range(0, len(u), 2):
            node_num = node_mapping_rev[u_node_num]
            # print('Node {}: ({}, {})'.format(node_num, u[x], u[x + 1]), 'Force: ', F[x], F[x + 1])
            print(f'Node {node_num}:', end=' | ')
            print(f'X Stiff {F[x]} (N) / {u[x]} (m)', end=' | ')
            print(f'Y Stiff {F[x + 1]} (N) / {u[x + 1]} (m)')
            u_node_num += 1

    # ------------------------------------------
    # Assemble Stiffness
    # ------------------------------------------

    # 1. Validate no very large displacements indicative of instability
    x_range = max([x for x, y in all_nodes]) - min([x for x, y in all_nodes])
    y_range = max([y for x, y in all_nodes]) - min([y for x, y in all_nodes])
    x_large_displacements = [x for x in u[mask_x] if abs(x) > (x_range * 10.0)]
    y_large_displacements = [y for y in u[mask_y] if abs(y) > (y_range * 10.0)]
    if len(x_large_displacements) > 0 or len(y_large_displacements) > 0:
        return 0

    # 2. Calculate stiffness_old for each node that has a load
    stiffness_vals = []
    for idx, load in enumerate(flat_loads_used):
        if load != 0:
            dof_force = F[idx]
            dof_disp = u[idx]
            if dof_disp == 0:
                stiffness_vals.append(0)
            else:
                if DEBUG:
                    print('LOAD NODE: ', idx, 'Force: ', dof_force, 'Disp: ', dof_disp)
                stiffness_vals.append(dof_force / dof_disp)
    if len(stiffness_vals) == 0:
        return 0
    stiffness = sum(stiffness_vals)
    return stiffness

























