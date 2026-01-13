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


import random
import textwrap
from itertools import combinations
import math
import numpy as np






def convert(problem, orig_rep):
    problem['nodes'] = sort_nodes(problem)
    nodes = problem['nodes']
    bit_members = get_bit_members(problem)

    bit_list = None
    bit_str = ''
    node_idx_pairs = None
    node_coords = None

    # Depending on representation, convert to bit list
    if isinstance(orig_rep, str):
        bit_str = orig_rep
        bit_list = []
        for char in bit_str:
            bit_list.append(int(char))
    elif isinstance(orig_rep, list):
        first_element = orig_rep[0]
        if isinstance(first_element, int):
            bit_list = orig_rep
        elif isinstance(first_element, tuple) or isinstance(first_element, list):
            ff_element = first_element[0]
            if isinstance(ff_element, int):
                node_idx_pairs = orig_rep
                bit_list = []
                for bm in bit_members:
                    if contains_pair(bm, node_idx_pairs):
                        bit_list.append(1)
                    else:
                        bit_list.append(0)
            else:
                node_coords = orig_rep
                # Convert to node index pairs
                node_idx_pairs = []
                for node_pair in node_coords:
                    idx_pair = []
                    for coord_pair in node_pair:
                        idx_pair.append(gcoords_to_node_idx(coord_pair, nodes))
                    node_idx_pairs.append(idx_pair)
                # Convert to bit list
                bit_list = []
                for bm in bit_members:
                    if contains_pair(bm, node_idx_pairs):
                        bit_list.append(1)
                    else:
                        bit_list.append(0)

    if bit_str == '':
        bit_str = ''.join([str(bit) for bit in bit_list])
    if node_idx_pairs is None:
        node_idx_pairs = []
        for idx, bit in enumerate(bit_list):
            if bit == 1:
                node_idx_pairs.append(bit_members[idx])
    if node_coords is None:
        node_coords = []
        for pair in node_idx_pairs:
            node_coords.append([nodes[pair[0]], nodes[pair[1]]])

    # print('Bit List:', bit_list, len(bit_list))
    # print('Bit Str:', bit_str)
    # print('Node Index Pairs:', node_idx_pairs)
    # print('Node Coords:', node_coords)

    return bit_list, bit_str, node_idx_pairs, node_coords

def sort_nodes(problem):
    # 1. Create a range of indices [0, 1, 2, ... N-1]
    indices = range(len(problem['nodes']))

    # 2. Sort the INDICES based on the values in problem['nodes']
    # We look up the node at index 'i' to determine the sort order
    sorted_indices = sorted(indices, key=lambda i: problem['nodes'][i])

    # 3. Apply this new order to 'nodes'
    problem['nodes'] = [problem['nodes'][i] for i in sorted_indices]

    # 4. Apply this new order to 'nodes_dof'
    problem['nodes_dof'] = [problem['nodes_dof'][i] for i in sorted_indices]

    # 5. Apply this new order to each load condition list
    # We iterate over every load case and reorder the inner list
    problem['load_conds'] = [
        [case[i] for i in sorted_indices]
        for case in problem['load_conds']
    ]
    return problem['nodes']

def get_bit_list_from_node_seq(problem, node_seq):
    node_idx_pairs = set()
    for i in range(len(node_seq) - 1):
        curr_node = node_seq[i]
        next_node = node_seq[i + 1]
        if curr_node == next_node:
            continue
        n1 = min(curr_node, next_node)
        n2 = max(curr_node, next_node)
        node_idx_pairs.add((n1, n2))
    node_idx_pairs = [list(x) for x in node_idx_pairs]
    bit_list, bit_str, node_idx_pairs, node_coords = convert(problem, node_idx_pairs)
    return bit_list

def gcoords_to_node_idx(coords, nodes):
    for idx, node in enumerate(nodes):
        if coords[0] == node[0] and coords[1] == node[1] and coords[2] == node[2]:
            return idx
    raise ValueError('Node not found in nodes')

def get_bit_members(problem):
    nodes = problem['nodes']
    bit_members = []
    for idx, node in enumerate(nodes):
        for idx2, node2 in enumerate(nodes):
            if idx2 > idx:
                bit_members.append((idx, idx2))
    return bit_members

def get_bit_coords(problem):
    nodes = sort_nodes(problem)
    bit_coords = []
    for idx, node in enumerate(nodes):
        for idx2, node2 in enumerate(nodes):
            if idx2 > idx:
                bit_coords.append((nodes[idx], nodes[idx2]))
    return bit_coords

def get_num_bits(problem):
    num_nodes = len(problem['nodes'])
    return int(num_nodes * (num_nodes - 1) / 2)

def contains_pair(pair, pairs):
    for p in pairs:
        if equate_pairs(pair, p) is True:
            return True
    return False

def equate_pairs(node1, node2):
    if node1[0] in node2 and node1[1] in node2:
        return True
    return False

def get_load_nodes(load_conds):
    load_nodes = set()
    for idx, node_load in enumerate(load_conds):
        if node_load[0] != 0 or node_load[1] != 0 or node_load[2] != 0:
            load_nodes.add(idx)
    return list(load_nodes)

def get_edge_nodes(problem):
    nodes = problem['nodes']
    min_x = min([x[0] for x in nodes])
    max_x = max([x[0] for x in nodes])
    min_y = min([x[1] for x in nodes])
    max_y = max([x[1] for x in nodes])
    min_z = min([x[2] for x in nodes])
    max_z = max([x[2] for x in nodes])
    edge_indices = []
    for idx, n in enumerate(nodes):
        if n[0] in [min_x, max_x] or n[1] in [min_y, max_y] or n[2] in [min_z, max_z]:
            edge_indices.append(idx)
    return edge_indices

def get_all_fixed_nodes(problem):
    nodes_dof = problem['nodes_dof']
    fixed_nodes = set()
    for idx, node_dof in enumerate(nodes_dof):
        if 0 in node_dof:
            fixed_nodes.add(idx)
    return list(fixed_nodes)

def get_fully_fixed_nodes(problem):
    nodes_dof = problem['nodes_dof']
    static_nodes = set()
    for idx, node_dof in enumerate(nodes_dof):
        if 1 not in node_dof:
            static_nodes.add(idx)
    return list(static_nodes)

def get_partially_fixed_nodes(problem):
    nodes_dof = problem['nodes_dof']
    fixed_nodes = set()
    for idx, node_dof in enumerate(nodes_dof):
        if 0 in node_dof and 1 in node_dof:
            fixed_nodes.add(idx)
    return list(fixed_nodes)

def get_free_nodes(problem):
    nodes_dof = problem['nodes_dof']
    free_nodes = set()
    for idx, node_dof in enumerate(nodes_dof):
        if 0 not in node_dof:
            free_nodes.add(idx)
    return list(free_nodes)

def get_used_nodes(problem, design_rep):
    bit_list, bit_str, node_idx_pairs, node_coords = convert(problem, design_rep)
    used_nodes = set()
    for pair in node_idx_pairs:
        for node in pair:
            used_nodes.add(node)
    return list(used_nodes)

def get_design_text(problem, design_rep):
    bit_list, bit_str, node_idx_pairs, node_coords = convert(problem, design_rep)
    reps = [
        0,  # Bit List
        # 1,  # Node Index Pairs
        # 2,  # Node Coords
    ]
    rand_rep = random.choice(reps)
    if rand_rep == 0:
        str_members = [str(x) for x in bit_list]
        design_text = ''.join(str_members)
        design_text = '[' + design_text + ']'
    elif rand_rep == 1:
        design_text = str(node_idx_pairs)
    elif rand_rep == 2:
        design_text = str(node_coords)

    return design_text

def get_node_connections(problem, design_rep, node_idx):
    bit_list, bit_str, node_idx_pairs, node_coords = convert(problem, design_rep)
    connections = []
    for pair in node_idx_pairs:
        if node_idx in pair:
            connections.append(pair)
    return connections

def calc_node_dist(node1, node2):
    return math.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2 + (node1[2] - node2[2]) ** 2)

def is_right_triangle(node1, node2, node3):
    sides = [((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2 + (node1[2] - node2[2]) ** 2),
             ((node2[0] - node3[0]) ** 2 + (node2[1] - node3[1]) ** 2 + (node2[2] - node3[2]) ** 2),
             ((node3[0] - node1[0]) ** 2 + (node3[1] - node1[1]) ** 2 + (node3[2] - node1[2]) ** 2)]
    sides.sort()
    dists = [calc_node_dist(node1, node2), calc_node_dist(node2, node3), calc_node_dist(node1, node3)]
    dists.sort()
    if abs(dists[0] - dists[1]) > 1e-6:
        return False
    else:
        return True

# ------------------------------
# Sampling
# ------------------------------

def random_sample_1(problem):  # Random bit list
    num_bits = get_num_bits(problem)
    return [random.choice([0, 1]) for _ in range(num_bits)]

# ------------------------------
# Overlap Score
# ------------------------------

class Point3D:
    def __init__(self, x, y, z):
        self.coords = np.array([x, y, z], dtype=float)

def calculate_overlap_score(problem, design_rep, tol=1e-9):
    # Standardize input (assuming 3D coordinates are returned)
    _, _, node_idx_pairs, node_coords_pairs = convert(problem, design_rep)

    crossing_violations = 0
    collinear_violations = 0
    num_members = len(node_idx_pairs)

    for i in range(num_members):
        for j in range(i + 1, num_members):
            idx1 = node_idx_pairs[i]
            idx2 = node_idx_pairs[j]

            # Shared node check
            shared = set(idx1).intersection(set(idx2))
            has_shared_node = len(shared) > 0

            # Coordinates as numpy arrays for vector math
            p1 = np.array(node_coords_pairs[i][0])
            q1 = np.array(node_coords_pairs[i][1])
            p2 = np.array(node_coords_pairs[j][0])
            q2 = np.array(node_coords_pairs[j][1])

            if has_shared_node:
                # Same logic as 2D: Check if they overlap in direction
                shared_idx = list(shared)[0]
                shared_pt = p1 if idx1[0] == shared_idx else q1
                tail1 = q1 if idx1[0] == shared_idx else p1
                tail2 = q2 if idx2[0] == shared_idx else p2

                v1 = tail1 - shared_pt
                v2 = tail2 - shared_pt

                # Normalize and check dot product
                unit_v1 = v1 / np.linalg.norm(v1)
                unit_v2 = v2 / np.linalg.norm(v2)

                # If dot product is ~1, they overlap in the same direction
                if np.dot(unit_v1, unit_v2) > (1 - tol):
                    collinear_violations += 1
            else:
                # 3D Segment-to-Segment Distance Check
                dist, is_parallel = segment_distance_3d(p1, q1, p2, q2)

                if dist < tol:
                    if is_parallel:
                        collinear_violations += 1
                    else:
                        crossing_violations += 1

    return {
        'score': crossing_violations + collinear_violations,
        'crossings': crossing_violations,
        'collinear': collinear_violations
    }

def segment_distance_3d(p1, q1, p2, q2):
    """
    Calculates the shortest distance between two 3D line segments.
    Based on the algorithm by Dan Sunday.
    """
    u = q1 - p1
    v = q2 - p2
    w = p1 - p2

    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w)
    e = np.dot(v, w)

    D = a * c - b * b
    sc, tc = 0, 0
    is_parallel = D < 1e-12

    if is_parallel:
        # Lines are parallel
        sc = 0.0
        tc = d / b if b > c else e / c
    else:
        # Get the closest points on the infinite lines
        sc = (b * e - c * d) / D
        tc = (a * e - b * d) / D

    # Clamp sc and tc to [0, 1] to stay within the segments
    sc = np.clip(sc, 0, 1)
    tc = np.clip(tc, 0, 1)

    # Shortest distance vector
    closest_dist_vec = w + (sc * u) - (tc * v)
    distance = np.linalg.norm(closest_dist_vec)

    return distance, is_parallel

# ------------------------------
# Angles
# ------------------------------

def get_bit_angles_3d(problem):
    """
    Calculates the 3D orientation angles (Azimuth and Elevation) for
    each truss member.
    """
    sort_nodes(problem)
    bit_coords = get_bit_coords(problem)  # List of pairs: [( (x1,y1,z1), (x2,y2,z2) ), ...]

    angles = []

    for (node1, node2) in bit_coords:
        dx = node2[0] - node1[0]
        dy = node2[1] - node1[1]
        dz = node2[2] - node1[2]

        # 1. Calculate Horizontal (Azimuth) Angle in XY plane
        # atan2 handles quadrants and vertical lines (dx=0) automatically
        azimuth_rad = np.arctan2(dy, dx)
        azimuth_deg = np.degrees(azimuth_rad)

        # 2. Calculate Vertical (Elevation) Angle
        # L_xy is the length of the projection on the XY plane
        L_xy = np.sqrt(dx ** 2 + dy ** 2)

        # Elevation is the angle relative to the XY plane
        elevation_rad = np.arctan2(dz, L_xy)
        elevation_deg = np.degrees(elevation_rad)

        # Normalize angles to your preference (e.g., 0-180 or 0-360)
        # Here we use standard atan2 ranges:
        # Azimuth: [-180, 180], Elevation: [-90, 90]
        angles.append((azimuth_deg, elevation_deg))

    return angles



# ------------------------------
# Visualization
# ------------------------------
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D # Necessary for 3D projection
import numpy as np
import os
import textwrap

def viz3d(problem, design_rep, f_name=None, optional_text=None, base_dir=None):
    """
    Visualizes a 3D truss design with multiple load conditions using matplotlib.
    """
    n_loads = len(problem['load_conds'])
    load_conds = problem['load_conds']

    # Calculate number of rows and columns for subplots
    n_loads_viz = n_loads
    cols = int(np.ceil(np.sqrt(n_loads_viz)))
    rows = int(np.ceil(n_loads_viz / cols))

    # Adjust figure size for 3D subplots (they generally need more space)
    fig = plt.figure(figsize=(5 * cols, 5 * rows), dpi=150)
    gs = gridspec.GridSpec(rows, cols)

    # --- Pre-calculate Geometry Limits for consistent aspect ratio ---
    nodal_locations = np.array(problem['nodes'])
    x_nodes = nodal_locations[:, 0]
    y_nodes = nodal_locations[:, 1]
    z_nodes = nodal_locations[:, 2]

    max_range = np.array([x_nodes.max() - x_nodes.min(),
                          y_nodes.max() - y_nodes.min(),
                          z_nodes.max() - z_nodes.min()]).max() / 2.0

    mid_x = (x_nodes.max() + x_nodes.min()) * 0.5
    mid_y = (y_nodes.max() + y_nodes.min()) * 0.5
    mid_z = (z_nodes.max() + z_nodes.min()) * 0.5

    # Define arrow scaling based on structure size
    arrow_len_scale = max_range * 0.2
    arrow_head_scale = arrow_len_scale * 0.3

    # --- Main Loop over Load Conditions ---
    for l_idx, load_cond in enumerate(load_conds):
        row = l_idx // cols
        col = l_idx % cols

        # Create 3D axis
        ax = fig.add_subplot(gs[row, col], projection='3d')

        # Get connectivity info
        # Assuming convert returns coordinates in 3D pairs: [[[x1,y1,z1],[x2,y2,z2]], ...]
        _, _, node_idx_pairs, _ = convert(problem, design_rep)

        # 1. Plotting the truss members
        for (start_idx, end_idx) in node_idx_pairs:
            p1 = nodal_locations[start_idx]
            p2 = nodal_locations[end_idx]
            # Plot line in 3D: plot([x1, x2], [y1, y2], [z1, z2])
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'k-', linewidth=1, zorder=1)

        # 2. Plotting the nodes and Boundary Conditions
        for i, node_loc in enumerate(nodal_locations):
            x, y, z = node_loc

            is_restrained = False
            if 'nodes_dof' in problem:
                # Assuming nodes_dof is now [bc_x, bc_y, bc_z], 0=fixed, 1=free
                node_dof = problem['nodes_dof'][i]

                if any(dof == 0 for dof in node_dof): is_restrained = True

                # Draw restraint arrows (red) using quiver
                # We draw short, thick arrows pointing *at* the node to indicate fixity
                if node_dof[0] == 0:  # X restrained
                    ax.quiver(x - arrow_len_scale, y, z, arrow_len_scale, 0, 0,
                              color='red', arrow_length_ratio=arrow_head_scale, linewidth=2, zorder=5)
                if node_dof[1] == 0:  # Y restrained
                    ax.quiver(x, y - arrow_len_scale, z, 0, arrow_len_scale, 0,
                              color='red', arrow_length_ratio=arrow_head_scale, linewidth=2, zorder=5)
                if node_dof[2] == 0:  # Z restrained
                    ax.quiver(x, y, z - arrow_len_scale, 0, 0, arrow_len_scale,
                              color='red', arrow_length_ratio=arrow_head_scale, linewidth=2, zorder=5)

            # Plot node marker based on restraint status
            if is_restrained:
                ax.scatter(x, y, z, c='r', marker='s', s=50, zorder=10)
            else:
                ax.scatter(x, y, z, c='b', marker='o', s=30, zorder=10)

            # Annotate Node Index (slightly offset for readability)
            ax.text(x + max_range * 0.02, y + max_range * 0.02, z + max_range * 0.02,
                    f'{i}', fontsize=9, zorder=20)

        # 3. Plot Node Loads for current condition
        # Assuming load_cond is list of [Fx, Fy, Fz]
        for i, forces in enumerate(load_cond):
            fx, fy, fz = forces
            if fx == 0 and fy == 0 and fz == 0: continue

            x, y, z = nodal_locations[i]

            # Normalize force vector for visualization length
            f_mag = np.sqrt(fx ** 2 + fy ** 2 + fz ** 2)
            u, v, w = (fx / f_mag, fy / f_mag, fz / f_mag)

            # Draw Load Vector (green) starting AT the node
            ax.quiver(x, y, z, u * arrow_len_scale, v * arrow_len_scale, w * arrow_len_scale,
                      color='green', arrow_length_ratio=0.3, linewidth=2, zorder=15)

            # Label force magnitude at tip of arrow
            ax.text(x + u * arrow_len_scale * 1.1,
                    y + v * arrow_len_scale * 1.1,
                    z + w * arrow_len_scale * 1.1,
                    f'{f_mag:.1f}N', color='green', fontsize=8, ha='center')

        # --- 3D Formatting ---
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Load Condition {l_idx}')

        # CRITICAL: Force equal aspect ratio in 3D view
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Optional: set a default camera view for better initial perspective
        ax.view_init(elev=20., azim=-45)

        # --- Global Text and Saving (Outside Loop) ---
    # # Assuming get_design_metrics works the same way
    # try:
    #     design_metrics = get_design_metrics(problem, design_rep)
    #     wrapped_lines = []
    #     for paragraph in design_metrics.split('\n'):
    #         wrapped_lines.extend(textwrap.wrap(paragraph, 60))
    #     wrapped_text = '\n'.join(wrapped_lines)
    #
    #     # Place text box generally in the bottom left area of the whole figure
    #     plt.figtext(0.02, 0.02, wrapped_text, ha='left', va='bottom', fontsize=9,
    #                 bbox=dict(facecolor='grey', alpha=0.1))
    # except NameError:
    #     print("Warning: get_design_metrics not found, skipping text box.")

    if optional_text:
        plt.figtext(0.5, 0.98, optional_text, ha='center', va='top', fontsize=12,
                    bbox=dict(facecolor='yellow', alpha=0.2))

    plt.tight_layout()

    # Save parameters
    # Note: Replace 'config.plots_dir' with your actual path variable if needed
    save_dir = base_dir if base_dir else '.'
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    if f_name is None:
        num_files = len(
            [name for name in os.listdir(save_dir) if name.startswith('truss_3d_') and name.endswith('.png')])
        f_name = f'truss_3d_{num_files}.png'

    full_path = os.path.join(save_dir, f_name)
    plt.savefig(full_path, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"3D Visualization saved to: {full_path}")



# ------------------------------
# Testing
# ------------------------------

if __name__ == '__main__':

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
                (1, 1, 1), (1, 1, 1),
                (1, 1, 1), (1, 1, 1),
            ],
            'load_conds': [ # Encodes the loads applied to each node in each direction (newtons)
                [  # Multiple load conditions can be specified
                    (0, 0, 0), (0, 0, 0),
                    (0, 0, 0), (0, 0, 0),
                    (0, 0, 0), (0, 0, 0),
                    (0, 0, 0), (1, 0, 0),
                ]
            ],
            'member_radii': 0.1,         # Radii is in meters
            'youngs_modulus': 1.8162e6,  # Material youngs modulus is in pascales (N/m^2)
    }

    d1_sample = random_sample_1(problem)

    viz3d(problem, d1_sample, 'd1_sample.png')




