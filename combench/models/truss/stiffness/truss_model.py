import numpy as np
import math

# from combench.models.truss.stiffness2.formK import fullK as getK
from combench.models.truss.stiffness.formK import formK as getK

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

def eval_load_cond(problem, design_rep, load_conds, verbose2=False):
    extra_info = {}
    if DEBUG:
        print('\n\n----------- Evaluating Load Condition -----------')
    if problem is None:
        raise ValueError('Problem input is None')
    for expected_key in ['nodes', 'nodes_dof', 'member_radii', 'youngs_modulus']:
        if expected_key not in problem:
            raise ValueError(f"Expected key '{expected_key}' not found in problem")

    # ------------------------------------------
    # Design Validation
    # ------------------------------------------

    # 1. Validate design has at least one member
    bit_list, bit_str, node_idx_pairs, node_coords = rep.convert(problem, design_rep)
    if sum(bit_list) == 0:
        extra_info['Error'] = 'No members in design'
        return 0, extra_info
    load_nodes = rep.get_load_nodes(load_conds)
    used_nodes_idx = rep.get_used_nodes(problem, bit_list)
    fixed_nodes = rep.get_all_fixed_nodes(problem)
    if DEBUG:
        print('Used Nodes:', used_nodes_idx)
        print('Fixed Nodes:', fixed_nodes)
        print('Load Nodes:', load_nodes)

    # 2. Validate node constraints
    free_used_nodes = [node for node in used_nodes_idx if node not in fixed_nodes]
    for node_idx in free_used_nodes:
        node_connections = rep.get_node_connections(problem, design_rep, node_idx)
        if DEBUG:
            print('Node', node_idx, 'connections:', node_connections)
        if len(node_connections) <= 1:
            extra_info['Error'] = 'Node {} has less than 2 connections'.format(node_idx)
            return 0, extra_info

    # 3. Validate all nodes with loads are used
    for ln in load_nodes:  # All load nodes with loads
        if ln not in used_nodes_idx:
            extra_info['Error'] = 'Load node {} not used'.format(ln)
            return 0, extra_info

    # 4. Validate at least one fixed node is used
    fixed_node_used = False
    for fn in fixed_nodes:
        if fn in used_nodes_idx:
            fixed_node_used = True
            break
    if not fixed_node_used:
        extra_info['Error'] = 'No fixed nodes used'
        return 0, extra_info

    # ------------------------------------------
    # Stiffness Calculation
    # ------------------------------------------
    # - used_nodes: coordinates of used nodes
    all_nodes = deepcopy(problem['nodes'])

    # 1. Filter out unused nodes
    all_to_used_map = {}  # Maps index of problem nodes to index of used nodes
    used_nodes = []
    used_nodes_dof = []  # problem['nodes_dof']
    used_nodes_loads = []
    for idx, node in enumerate(all_nodes):
        if idx in used_nodes_idx:
            used_nodes.append(node)
            used_nodes_dof.append(problem['nodes_dof'][idx])
            used_nodes_loads.append(load_conds[idx])
            all_to_used_map[idx] = len(used_nodes) - 1
    used_to_all_map = {v: k for k, v in all_to_used_map.items()}
    if DEBUG:
        print('\nORIGINAL NODES:', all_nodes)
        print('USED NODES:', used_nodes)
        print('USED NODES DOF:', used_nodes_dof)
        print('USED NODES LOADS:', used_nodes_loads)

    # 2.1 Get used fixed nodes indices
    used_fixed_nodes_idx = []
    for fn in fixed_nodes:
        if fn in all_to_used_map:
            used_fixed_nodes_idx.append(all_to_used_map[fn])
    if DEBUG:
        print('USED FIXED NODES IDX:', used_fixed_nodes_idx)

    # 2.2 Get used load nodes indices
    used_load_nodes_idx = []
    for ln in load_nodes:
        if ln in all_to_used_map:
            used_load_nodes_idx.append(all_to_used_map[ln])
    if DEBUG:
        print('USED LOAD NODES IDX:', used_load_nodes_idx)

    # 2.3 Get design node_idx pairs for used nodes indices
    used_node_idx_pairs = []
    for idx_pair in node_idx_pairs:
        n1, n2 = idx_pair
        n1_used, n2_used = all_to_used_map[n1], all_to_used_map[n2]
        used_node_idx_pairs.append((n1_used, n2_used))

    # 3. Find the cross-sectional area of each used member
    m_rad = problem['member_radii']
    m_areas = np.array([np.pi * m_rad ** 2 for _ in range(len(used_node_idx_pairs))])
    if DEBUG:
        print('\nMember Radii:', m_rad)
        print('Member Areas:', m_areas)


    # 4. Add indices to nodes and members (elements)
    new_coords = []
    for idx, nc in enumerate(used_nodes):
        # new_coords.append([idx + 1, nc[0], nc[1]])
        new_coords.append([nc[0], nc[1]])
    used_nodes = np.array(new_coords)

    new_conns = []
    for idx, nc in enumerate(used_node_idx_pairs):
        # new_conns.append([idx + 1, nc[0], nc[1]])
        new_conns.append([nc[0], nc[1]])
    elements = np.array(new_conns) + 1

    # 5. Calculate the global stiffness_old matrix (check if singular)
    y_mod = problem['youngs_modulus']

    if DEBUG:
        print('\n----- STIFFNESS MATRIX INPUTS:')
        print('Nodes:', used_nodes)
        print('Elements:', elements)
        print('Areas:', m_areas)
        print('Youngs Modulus:', y_mod)
    K = getK(used_nodes, elements, m_areas, y_mod)  # Global stiffness_old matrix

    K_full = getK(np.array(all_nodes), np.array(node_idx_pairs)+1, m_areas, y_mod)

    K_det = np.linalg.det(K)
    if DEBUG:
        print('\nGlobal stiffness matrix:')
        print(K)
        print('\nGlobal stiffness matrix determinant:', K_det, np.isclose(K_det, 0))
    # if np.isclose(K_det, 0):
    #     return 0, extra_info

    # 6. Create a mask to filter out fixed nodes (working with used nodes only)
    # - 0: fixed, 1: free
    mask_fixed_nodes = []
    for node_dof in used_nodes_dof:
        mask_fixed_nodes.extend(list(node_dof))
    num_free = sum(mask_fixed_nodes)
    mask_fixed_nodes = np.array(mask_fixed_nodes).astype(bool)

    # 7. Create mask to filter out x and y nodes
    mask_y = (np.arange(1, len(mask_fixed_nodes) + 1) % 2).astype(bool)  # [1, 0, 1, 0, 1, 0, ...]
    mask_x = np.logical_not(mask_y).astype(bool)                         # [0, 1, 0, 1, 0, 1, ...]

    # 8. Flatten node loads (working with used nodes only)
    flat_loads = []
    flat_loads_used = []
    for node_load in used_nodes_loads:
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
        extra_info['Error'] = 'Linear algebra error solving for free node displacements'
        return 0, extra_info

    # 13. Construct full displacement vector with fixed nodes (working with used nodes only)
    u = np.zeros(2 * len(used_nodes))
    u[mask_fixed_nodes] = uf  # Set free node displacements

    # 14. Calculate all nodal forces with full displacement vector
    try:
        F = K @ u

        # print('\n--> evaluating truss design')
        # print('F:', F.shape[0])
        # print('u:', u.shape)
        # print('K:', K.shape)
        # exit(0)



    except Exception as e:
        if DEBUG:
            print('Linear alg error:', e)
        extra_info['Error'] = 'Linear algebra error calculating nodal forces'
        return 0, extra_info

    # ------------------------------------------
    # DEBUG SOLVER
    # ------------------------------------------


    F_full = np.zeros((36,))
    u_full = np.zeros((36,))
    if verbose2 is True:
        u_node_num = 0  # in used nodes frame
        for x in range(0, len(u), 2):
            node_num = used_to_all_map[u_node_num]
            F_full[node_num * 2] = F[x]
            F_full[node_num * 2 + 1] = F[x + 1]
            u_full[node_num * 2] = u[x]
            u_full[node_num * 2 + 1] = u[x + 1]
            u_node_num += 1

        extra_info['F_full'] = F_full
        extra_info['u_full'] = u_full
        extra_info['K_full'] = K_full



    if DEBUG:
        u_node_num = 0  # in used nodes frame
        for x in range(0, len(u), 2):
            node_num = used_to_all_map[u_node_num]
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
        extra_info['Error'] = 'Large displacements indicative of critical failure'
        return 0, extra_info

    # 2. Calculate stiffness for each node that has a load
    # stiffness_vals = []
    # for idx, load in enumerate(flat_loads_used):
    #     full_idx = used_to_all_map[idx//2]
    #     if idx % 2 == 0:
    #         title = 'Node' + str(full_idx) + ' X'
    #     else:
    #         title = 'Node' + str(full_idx) + ' Y'
    #     if load != 0:
    #         dof_force = F[idx]
    #         dof_disp = u[idx]
    #         extra_info[title] = f'{dof_force:.2e} N | {dof_disp:.2e} m'
    #         if dof_disp == 0:
    #             stiffness_vals.append(0)
    #         else:
    #             if DEBUG:
    #                 print('LOAD NODE:', full_idx, 'Force: ', dof_force, 'Disp: ', dof_disp)
    #             stiffness_vals.append(dof_force / dof_disp)
    # if len(stiffness_vals) == 0:
    #     extra_info['Error'] = 'No loads applied'
    #     return 0, extra_info
    # stiffness = sum(stiffness_vals)

    # stiffness, extra_info = get_stiffness_old(flat_loads_used, F, u, used_to_all_map, extra_info)

    stiffness, extra_info = get_stiffness(flat_loads_used, F, u, used_to_all_map, extra_info, verbose2)




    return stiffness, extra_info




def get_stiffness(flat_loads_used, F, u, idx_map, extra_info, verbose2=False):
    node_stiffness = []
    node_dists = []
    node_forces = []

    # Iterates over each node that has a load applied
    # - For the cantilever problem, there will only be one node with a load applied
    for idx in range(0, len(flat_loads_used), 2):
        node_idx = idx // 2
        node_title = 'Node ' + str(idx_map[node_idx])
        nx_idx, ny_idx = idx, idx + 1
        load_x = flat_loads_used[nx_idx]
        load_y = flat_loads_used[ny_idx]

        node_text = ''
        if load_x != 0 and load_y != 0:
            f = get_magnitude(F[nx_idx], F[ny_idx])
            u = get_magnitude(u[nx_idx], u[ny_idx])
        elif load_x != 0:
            f = F[nx_idx]
            u = u[nx_idx]
        elif load_y != 0:
            f = F[ny_idx]
            u = u[ny_idx]
        else:  # No load applied
            continue
        node_text = f'{f:.2e} N | {u:.2e} m'
        extra_info[node_title] = node_text

        if u == 0:
            node_stiffness.append(0)
        else:
            node_stiffness.append(f / u)
        node_dists.append(abs(u))
        node_forces.append(abs(f))


    if verbose2 is True:
        extra_info['node_dists'] = node_dists
        extra_info['node_forces'] = node_forces

    # Metric 1: Total Stiffness
    t_dist = sum(node_dists)
    if t_dist == 0:
        t_stiff = 0
    else:
        t_stiff = sum(node_forces) / t_dist

    # Metric 2: Node-wise Stiffness
    n_stiff = sum(node_stiffness)

    return t_stiff, extra_info



def get_magnitude(x_comp, y_comp):
    mag = (x_comp**2) + (y_comp**2)
    return math.sqrt(mag)







def get_stiffness_old(flat_loads_used, F, u, used_to_all_map, extra_info):
    stiffness_vals = []
    for idx, load in enumerate(flat_loads_used):
        full_idx = used_to_all_map[idx // 2]
        if idx % 2 == 0:
            title = 'Node' + str(full_idx) + ' X'
        else:
            title = 'Node' + str(full_idx) + ' Y'
        if load != 0:
            dof_force = F[idx]
            dof_disp = u[idx]
            extra_info[title] = f'{dof_force:.2e} N | {dof_disp:.2e} m'
            if dof_disp == 0:
                stiffness_vals.append(0)
            else:
                if DEBUG:
                    print('LOAD NODE:', full_idx, 'Force: ', dof_force, 'Disp: ', dof_disp)
                stiffness_vals.append(dof_force / dof_disp)
    if len(stiffness_vals) == 0:
        extra_info['Error'] = 'No loads applied'
        return 0, extra_info
    stiffness = sum(stiffness_vals)
    return stiffness, extra_info













