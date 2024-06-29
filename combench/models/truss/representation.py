import random
import textwrap

""" 
----- Design Representations -----

 - Bit List
 - Bit Str
 - List of node index pairs: [
          [0, 1],
          [0, 2]
    ]
 - List of node coordinates: [
        [ [0, 1], [1, 2] ] ,
        [ [0, 1], [1, 3] ]
    ]


----- Problem Data Structure -----

 - Nodes must always be ordered by x-coordinate, then y-coordinate
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


def sort_nodes(problem):
    nodes = sorted(problem['nodes'], key=lambda x: (x[0], x[1]))
    problem['nodes'] = nodes
    return nodes


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


def gcoords_to_node_idx(coords, nodes):
    for idx, node in enumerate(nodes):
        if coords[0] == node[0] and coords[1] == node[1]:
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
        if node_load[0] != 0 or node_load[1] != 0:
            load_nodes.add(idx)
    return list(load_nodes)


def get_edge_nodes(problem):
    nodes = problem['nodes']
    min_x = min([x[0] for x in nodes])
    max_x = max([x[0] for x in nodes])
    min_y = min([x[1] for x in nodes])
    max_y = max([x[1] for x in nodes])
    edge_indices = []
    for idx, n in enumerate(nodes):
        if n[0] in [min_x, max_x] or n[1] in [min_y, max_y]:
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


# ------------------------------
# Sampling
# ------------------------------
from itertools import combinations


def random_sample_1(problem):  # Random bit list
    num_bits = get_num_bits(problem)
    return [random.choice([0, 1]) for _ in range(num_bits)]


def random_sample_2(problem, small=False):  # Random number of 1s
    num_bits = get_num_bits(problem)
    rand_bit_members = [0 for x in range(num_bits)]
    num_1s = random.choice(range(num_bits - 1)) + 1
    if small is True:
        max_bits = num_bits // 3
        num_1s = min(num_1s, max_bits)
    indices = random.sample(range(num_bits), num_1s)
    for i in indices:
        rand_bit_members[i] = 1
    return rand_bit_members


def random_sample_3(problem):  # Random number of nodes with connections
    nodes = problem['nodes']
    num_nodes = len(nodes)
    num_sample_nodes = random.randint(2, num_nodes)
    node_indices = [idx for idx in range(len(nodes))]
    sample_nodes = random.sample(node_indices, num_sample_nodes)
    bit_members = get_bit_members(problem)
    bit_list = []
    for bm in bit_members:
        if bm[0] in sample_nodes and bm[1] in sample_nodes:
            bit_list.append(random.choice([0, 1]))
        else:
            bit_list.append(0)
    if 1 not in bit_list:
        bit_list = []
        for bm in bit_members:
            if bm[0] in sample_nodes and bm[1] in sample_nodes:
                bit_list.append(1)
            else:
                bit_list.append(0)
    return bit_list


def grid_design_sample(problem):
    nodes = problem['nodes']
    node_indices = [idx for idx in range(len(nodes))]
    node_pairs = []
    for i in node_indices:
        i_neighbors = get_node_neighbors(problem, i)
        print('Node Neighbors:', i, i_neighbors)
        for j in i_neighbors:
            if contains_pair((i, j), node_pairs) is False:
                n_pair = (min(i, j), max(i, j))
                node_pairs.append(n_pair)
    bit_list, bit_str, node_idx_pairs, node_coords = convert(problem, node_pairs)
    # bit_list = remove_overlapping_members(problem, bit_list)
    return bit_list


def get_tri_design(problem):
    nodes_objs = problem['nodes']
    nodes_idx = [idx for idx in range(len(nodes_objs))]
    edges = []
    for node1, node2, node3 in combinations(nodes_idx, 3):
        if (node1 == node2) or (node2 == node3) or (node1 == node3):
            continue
        if not is_right_triangle(nodes_objs[node1], nodes_objs[node2], nodes_objs[node3]):
            continue
        edge1 = (node1, node2)
        edge2 = (node2, node3)
        edge3 = (node1, node3)
        if not contains_pair(edge1, edges):
            edges.append(edge1)
        if not contains_pair(edge2, edges):
            edges.append(edge2)
        if not contains_pair(edge3, edges):
            edges.append(edge3)
    edges = list(edges)
    bit_list, bit_str, node_idx_pairs, node_coords = convert(problem, edges)
    return bit_list


def calc_node_dist(node1, node2):
    return math.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2)


def is_right_triangle(node1, node2, node3):
    sides = [((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2),
             ((node2[0] - node3[0]) ** 2 + (node2[1] - node3[1]) ** 2),
             ((node3[0] - node1[0]) ** 2 + (node3[1] - node1[1]) ** 2)]
    sides.sort()
    dists = [calc_node_dist(node1, node2), calc_node_dist(node2, node3), calc_node_dist(node1, node3)]
    dists.sort()
    if abs(dists[0] - dists[1]) > 1e-6:
        return False
    else:
        return True
    # return sides[0] + sides[1] == sides[2]


def get_node_neighbors(problem, node_idx):
    nodes = problem['nodes']
    node_indices = [idx for idx in range(len(nodes))]
    root_node = nodes[node_idx]

    # Get bit members
    diag_members = []
    diag_nodes = set()
    bit_members = get_bit_members(problem)
    bit_member_angles = get_bit_angles(problem)
    for bm, bma in zip(bit_members, bit_member_angles):
        if bma - 45.0 > 1e-6:
            if node_idx in bm:
                diag_members.append(bm)
                diag_nodes.add(bm[0] if bm[1] == node_idx else bm[1])

    neighbors = set()
    x_neighbors = [idx for idx in node_indices if (nodes[idx][1] == root_node[1] and nodes[idx][0] != root_node[0])]
    y_neighbors = [idx for idx in node_indices if (nodes[idx][0] == root_node[0] and nodes[idx][1] != root_node[1])]
    # diag_neighbors = [idx for idx in node_indices if abs(nodes[idx][0] - root_node[0]) == abs(nodes[idx][1] - root_node[1])]
    diag_neighbors = diag_nodes

    print('Diag Neighbors:', diag_neighbors)

    # sort to get the closest neighbors
    x_neighbors = sorted(x_neighbors, key=lambda x: abs(nodes[x][0] - root_node[0]))
    y_neighbors = sorted(y_neighbors, key=lambda x: abs(nodes[x][1] - root_node[1]))
    diag_neighbors = sorted(diag_neighbors, key=lambda x: abs(nodes[x][0] - root_node[0]))

    # print('X Neighbors:', x_neighbors)
    # print('Y Neighbors:', y_neighbors)

    min_x_dist = abs(nodes[x_neighbors[0]][0] - root_node[0])
    min_y_dist = abs(nodes[y_neighbors[0]][1] - root_node[1])

    # Get all x_neighbors with the distance min_x_dist
    tolerance = 1e-6
    x_neighbors = [idx for idx in x_neighbors if abs(nodes[idx][0] - root_node[0]) - min_x_dist < tolerance]
    y_neighbors = [idx for idx in y_neighbors if abs(nodes[idx][1] - root_node[1]) - min_y_dist < tolerance]

    neighbors.update(x_neighbors)
    neighbors.update(y_neighbors)
    neighbors.update(diag_neighbors)
    if node_idx in neighbors:
        neighbors.remove(node_idx)
    return list(neighbors)


# ------------------------------
# Remove Overlaps (OLD)
# ------------------------------
# - Horizontally or Vertically overlapping

def remove_overlapping_members_old(problem, design_rep):
    bit_list, bit_str, node_idx_pairs, node_coords = convert(problem, design_rep)
    overlapped_indices = find_overlapped_segments(node_coords)
    print('Over lapped indices:', overlapped_indices)
    print('Over lapped pairs:', [node_idx_pairs[i] for i in overlapped_indices])
    no_node_idx_pairs = [node_idx_pairs[i] for i in range(len(node_idx_pairs)) if i not in overlapped_indices]
    bit_list, bit_str, node_idx_pairs, node_coords = convert(problem, no_node_idx_pairs)
    return bit_list

def find_overlapped_segments(segments):
    """Find indices of line segments that are completely overlapped by another segment."""
    overlapped_indices = []
    n = len(segments)
    for i in range(n):
        for j in range(n):
            if i != j and is_segment_overlapped(segments[i], segments[j]):
                overlapped_indices.append(i)
                break
    return overlapped_indices


def is_segment_overlapped(seg1, seg2):
    # return check_overlap(seg1, seg2)
    """Check if line segment seg1 is completely overlapped by line segment seg2."""
    (x1, y1), (x2, y2) = seg1
    (x3, y3), (x4, y4) = seg2
    return (is_point_on_segment(x1, y1, x3, y3, x4, y4) and
            is_point_on_segment(x2, y2, x3, y3, x4, y4))


def is_point_on_segment(px, py, x1, y1, x2, y2):
    """Check if point (px, py) lies on the line segment from (x1, y1) to (x2, y2)."""
    if min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2):
        if (x2 - x1) * (py - y1) == (y2 - y1) * (px - x1):
            return True
    return False


# ------------------------------
# Remove Overlaps (NEW)
# ------------------------------

def remove_overlapping_members(problem, design_rep):
    bit_list, bit_str, node_idx_pairs, node_coords_pairs = convert(problem, design_rep)

    # Sort node coord and index pairs by length
    node_coords_idx_pairs = list(zip(node_coords_pairs, node_idx_pairs))
    node_coords_idx_pairs = sorted(node_coords_idx_pairs, key=lambda x: calc_length(x[0]), reverse=True)
    node_coords_pairs, node_idx_pairs = zip(*node_coords_idx_pairs)

    overlapping_indices = set()
    removed_members = []
    for idx1, pair1 in enumerate(node_coords_pairs):
        m1 = node_idx_pairs[idx1]
        if idx1 in overlapping_indices:
            continue
        for idx2, pair2 in enumerate(node_coords_pairs):
            if idx2 <= idx1:
                continue
            m2 = node_idx_pairs[idx2]
            int_exists, int_len = doIntersect(pair1, pair2)
            if int_exists is True and int_len > 0.0:
                p1_len = calc_length(pair1)
                p2_len = calc_length(pair2)
                if p1_len > p2_len:
                    overlapping_indices.add(idx1)
                    removed_members.append(m1)
                    break
                else:
                    overlapping_indices.add(idx2)
                    removed_members.append(m2)
    no_overlap_node_coord_pairs = []
    # print('REMOVED MEMBERS:', removed_members)
    for idx, node_coords_pair in enumerate(node_coords_pairs):
        if idx not in overlapping_indices:
            no_overlap_node_coord_pairs.append(node_coords_pair)
    new_bit_list, new_bit_str, new_node_idx_pairs, new_node_coords = convert(problem, no_overlap_node_coord_pairs)
    return new_bit_list

def calc_length(node_coords_pair):
    n1_coords, n2_coords = node_coords_pair
    p, q = Point(*n1_coords), Point(*n2_coords)
    return ((q.x - p.x)**2 + (q.y - p.y)**2)**0.5


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def onSegment(p, q, r):
    if ( (q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and
           (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))):
        return True
    return False


def orientation(p, q, r):
    val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y))
    if (val > 0):
        return 1
    elif (val < 0):
        return 2
    else:
        return 0


def doIntersect(node_coords_pair_1, node_coords_pair_2):
    n1_coords, n2_coords = node_coords_pair_1
    p1, q1 = Point(*n1_coords), Point(*n2_coords)
    n3_coords, n4_coords = node_coords_pair_2
    p2, q2 = Point(*n3_coords), Point(*n4_coords)
    return _doIntersect(p1, q1, p2, q2)


def _doIntersect(p1, q1, p2, q2):
    def length(p, q):
        return ((q.x - p.x)**2 + (q.y - p.y)**2)**0.5

    def collinear_segment_length(p1, q1, p2, q2):
        # Calculate the overlapping segment's length
        if onSegment(p1, p2, q1) and onSegment(p1, q2, q1):
            # p2 and q2 are both within p1q1
            return length(p2, q2)
        elif onSegment(p2, p1, q2) and onSegment(p2, q1, q2):
            # p1 and q1 are both within p2q2
            return length(p1, q1)
        elif onSegment(p1, p2, q1):
            # p2 is within p1q1 and q1 is within p2q2
            return length(p2, q1)
        elif onSegment(p1, q2, q1):
            # q2 is within p1q1 and p1 is within p2q2
            return length(p1, q2)
        else:
            return 0.0

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # -----------------------------
    # General Case
    # -----------------------------

    if ((o1 != o2) and (o3 != o4)):
        # print(' - General Case: Non-collinear segments intersect')
        return False, 0.0

    # -----------------------------
    # Special Cases
    # -----------------------------
    # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
    if ((o1 == 0) and onSegment(p1, p2, q1)):
        # print(' - Case 1: p1 , q1 and p2 are collinear and p2 lies on segment p1q1')
        return True, collinear_segment_length(p1, q1, p2, q2)

    # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
    if ((o2 == 0) and onSegment(p1, q2, q1)):
        # print(' - Case 2: p1 , q1 and q2 are collinear and q2 lies on segment p1q1')
        return True, collinear_segment_length(p1, q1, p2, q2)

    # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
    if ((o3 == 0) and onSegment(p2, p1, q2)):
        # print(' - Case 3: p2 , q2 and p1 are collinear and p1 lies on segment p2q2')
        return True, collinear_segment_length(p1, q1, p2, q2)

    # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
    if ((o4 == 0) and onSegment(p2, q1, q2)):
        # print(' - Case 4: p2 , q2 and q1 are collinear and q1 lies on segment p2q2')
        return True, collinear_segment_length(p1, q1, p2, q2)

    # If none of the cases
    return False, 0.0


# ------------------------------
# Angles
# ------------------------------
import math
import numpy as np


def get_bit_angles(problem):
    sort_nodes(problem)
    bit_coords = get_bit_coords(problem)  # List pairs, where each pair holds the coordinates of two nodes
    angles = []
    for (node1, node2) in bit_coords:
        if node1[0] == node2[0]:  # X coords are the same, vertical
            angles.append(90.0)
            continue
        elif node1[1] == node2[1]:  # Y coords are the same, horizontal
            angles.append(0.0)
            continue

        y_delta = abs(node1[1] - node2[1])  # Opposite
        x_delta = abs(node1[0] - node2[0])  # Adjacent
        angle_rad = np.arctan(y_delta / x_delta)
        angle_deg = np.degrees(angle_rad)
        angles.append(angle_deg)
    return angles


def get_design_angle(problem, design_rep):
    bit_list, bit_str, node_idx_pairs, node_coords = convert(problem, design_rep)
    if 1 not in bit_list:
        return 0.0
    bit_angles = get_bit_angles(problem)
    # bit_list_no_overlaps = remove_overlapping_members(problem, design_rep)
    bit_list_no_overlaps = bit_list
    angles = []
    for idx, bit in enumerate(bit_list_no_overlaps):
        if bit == 1:
            angles.append(bit_angles[idx])
    ang = np.mean(angles)
    return ang


# ------------------------------
# Visualization
# ------------------------------
import matplotlib.pyplot as plt
import os
import config
from matplotlib.gridspec import GridSpec
import matplotlib.gridspec as gridspec


def viz(problem, design_rep, f_name=None, optional_text=None, base_dir=None):
    return viz2(problem, design_rep, f_name, optional_text, base_dir)


def viz1(problem, design_rep, f_name=None, optional_text=None, base_dir=None):
    gs = gridspec.GridSpec(1, 2)
    fig = plt.figure(figsize=(10, 5))  # default [6.4, 4.8], W x H  9x6, 12x8
    plt.subplot(gs[0, 0])
    # fig.suptitle('Design Edit Model', fontsize=16)

    bit_list, bit_str, node_idx_pairs, node_coords = convert(problem, design_rep)
    nodal_locations = problem['nodes']

    # x_range = max([x[0] for x in nodal_locations]) - min([x[0] for x in nodal_locations])
    # y_range = max([x[1] for x in nodal_locations]) - min([x[1] for x in nodal_locations])

    # Plotting the truss members
    for (start, end) in node_idx_pairs:
        x_coords = [nodal_locations[start][0], nodal_locations[end][0]]
        y_coords = [nodal_locations[start][1], nodal_locations[end][1]]
        plt.plot(x_coords, y_coords, 'black')  # Plot truss members as red lines

    # Get axis
    ax = plt.gca()
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Plotting the nodes
    for i, (x, y) in enumerate(nodal_locations):
        if 'nodes_dof' in problem:
            node_dof = problem['nodes_dof'][i]
            if node_dof[0] == 0 or node_dof[1] == 0:  # Node restrained in at least 1 dof
                plt.plot(x, y, 'ro')  # Plot nodes as red squares
                if node_dof[0] == 0:
                    arrow_start = [x, y]
                    arrow_end = [x + (x_range / 10), y]
                    ax.arrow(arrow_start[0], arrow_start[1], arrow_end[0] - arrow_start[0],
                             arrow_end[1] - arrow_start[1],
                             head_width=0.2, head_length=0.2, fc='red', ec='red', width=0.05, zorder=2)
                if node_dof[1] == 0:
                    arrow_start = [x, y]
                    arrow_end = [x, y + (y_range / 10)]
                    ax.arrow(arrow_start[0], arrow_start[1], arrow_end[0] - arrow_start[0],
                             arrow_end[1] - arrow_start[1],
                             head_width=0.2, head_length=0.2, fc='red', ec='red', width=0.05, zorder=2)
            else:
                plt.plot(x, y, 'bo')  # Plot nodes as blue circles
        else:
            plt.plot(x, y, 'bo')  # Plot nodes as blue circles
        plt.text(x, y, f'{i}', fontsize=12, ha='right')  # Annotate nodes with their index

    # Node Loads
    if 'load_conds' in problem:
        nodes_loads = problem['load_conds'][0]
        ax = plt.gca()  # Get the current axis
        # nodes_loads = problem['nodes_loads']
        for I, (load_x, load_y) in enumerate(nodes_loads):
            # print('LOAD:', I, load_x, load_y)
            if load_x != 0:
                arrow_start = [nodal_locations[I][0], nodal_locations[I][1]]
                if load_x > 0:
                    arrow_end = [arrow_start[0] + (x_range / 10), arrow_start[1]]
                else:
                    arrow_end = [arrow_start[0] - (x_range / 10), arrow_start[1]]
                ax.arrow(arrow_start[0], arrow_start[1], arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1],
                         head_width=0.2, head_length=0.2, width=0.05,
                         fc='green', ec='green', zorder=2)
                plt.text(arrow_end[0], arrow_end[1] + (y_range / 100), f'{load_x:.2f} N', fontsize=10,
                         ha='right')  # Annotate nodes with their index

            if load_y != 0:
                arrow_start = [nodal_locations[I][0], nodal_locations[I][1]]
                if load_y > 0:
                    arrow_end = [arrow_start[0], arrow_start[1] + (y_range / 10)]
                else:
                    arrow_end = [arrow_start[0], arrow_start[1] - (y_range / 10)]
                ax.arrow(arrow_start[0], arrow_start[1], arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1],
                         head_width=0.2, head_length=0.2, width=0.05,
                         fc='green', ec='green', zorder=2)
                plt.text(arrow_end[0], arrow_end[1] - (x_range / 100), f'{load_y:.2f} N', fontsize=10,
                         ha='right')  # Annotate nodes with their index

    # Adding optional text
    if optional_text is not None:
        plt.figtext(
            0.5, 0.05,
            optional_text,
            ha='center', va='bottom', fontsize=10,
            bbox=dict(facecolor='red', alpha=0.5)
        )
        plt.subplots_adjust(bottom=0.3)

        # plt.text(text_position[0], text_position[1], optional_text, fontsize=12, ha='center',
        #          transform=plt.gca().transAxes)

    # Setting the plot labels and title
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Truss Structure')
    plt.grid(True)
    plt.axis('equal')  # Ensure the aspect ratio is equal to avoid distortion

    # Design Text
    design_metrics = get_design_metrics(problem, design_rep)
    plt.figtext(
        0.5, 0.05,
        design_metrics,
        ha='left', va='bottom', fontsize=10,
        bbox=dict(facecolor='grey', alpha=0.5)
    )

    # Save the plot
    if base_dir is None:
        base_dir = config.plots_dir

    if f_name is None:
        num_files = len([name for name in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, name))])
        f_name = f'truss_{num_files}.png'
    plt.savefig(os.path.join(base_dir, f_name))
    plt.close('all')

def viz2(problem, design_rep, f_name=None, optional_text=None, base_dir=None):
    n_loads = len(problem['load_conds'])
    load_conds = problem['load_conds']
    # max_viz = 3
    # if n_loads > max_viz:
    #     n_loads = max_viz
    #     load_conds = load_conds[:max_viz]

    # Calculate number of rows and columns for subplots
    n_loads_extra = n_loads + 1
    cols = int(np.ceil(np.sqrt(n_loads_extra)))
    rows = int(np.ceil(n_loads_extra / cols))

    # Create GridSpec layout
    gs = gridspec.GridSpec(rows, cols)
    fig = plt.figure(figsize=(3.5 * cols, 3.5 * rows), dpi=150)  # Adjust the figure size based on rows and cols

    # gs = gridspec.GridSpec(2, n_loads)
    # fig = plt.figure(figsize=(5*n_loads, 5))  # default [6.4, 4.8], W x H  9x6, 12x8

    center_y = 0
    for l_idx, load_cond in enumerate([None] + load_conds):
        if l_idx == 0:
            continue


        # plt.subplot(gs[0, l_idx])
        row = l_idx // cols
        col = l_idx % cols
        plt.subplot(gs[row, col])

        if l_idx == 1:
            ax = plt.gca()
            pos = ax.get_position()  # Get the position of the subplot
            # x_center = (pos.x0 + pos.x1) / 2  # Calculate the center x position
            y_center = (pos.y0)  # + pos.y1) / 2  # Calculate the center y position


        bit_list, bit_str, node_idx_pairs, node_coords = convert(problem, design_rep)
        nodal_locations = problem['nodes']

        # x_range = max([x[0] for x in nodal_locations]) - min([x[0] for x in nodal_locations])
        # y_range = max([x[1] for x in nodal_locations]) - min([x[1] for x in nodal_locations])

        # Plotting the truss members
        for (start, end) in node_idx_pairs:
            x_coords = [nodal_locations[start][0], nodal_locations[end][0]]
            y_coords = [nodal_locations[start][1], nodal_locations[end][1]]
            plt.plot(x_coords, y_coords, 'black')  # Plot truss members as red lines

        # Get axis
        ax = plt.gca()
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x_range = x_max - x_min
        y_range = y_max - y_min

        # Plotting the nodes
        for i, (x, y) in enumerate(nodal_locations):
            if 'nodes_dof' in problem:
                node_dof = problem['nodes_dof'][i]
                if node_dof[0] == 0 or node_dof[1] == 0:  # Node restrained in at least 1 dof
                    plt.plot(x, y, 'ro')  # Plot nodes as red squares
                    if node_dof[0] == 0:
                        arrow_start = [x, y]
                        arrow_end = [x + (x_range / 10), y]
                        ax.arrow(arrow_start[0], arrow_start[1], arrow_end[0] - arrow_start[0],
                                 arrow_end[1] - arrow_start[1],
                                 head_width=0.2, head_length=0.2, fc='red', ec='red', width=0.05, zorder=2)
                    if node_dof[1] == 0:
                        arrow_start = [x, y]
                        arrow_end = [x, y + (y_range / 10)]
                        ax.arrow(arrow_start[0], arrow_start[1], arrow_end[0] - arrow_start[0],
                                 arrow_end[1] - arrow_start[1],
                                 head_width=0.2, head_length=0.2, fc='red', ec='red', width=0.05, zorder=2)
                else:
                    plt.plot(x, y, 'bo')  # Plot nodes as blue circles
            else:
                plt.plot(x, y, 'bo')  # Plot nodes as blue circles
            plt.text(x, y, f'{i}', fontsize=12, ha='right')  # Annotate nodes with their index

        # Node Loads
        if 'load_conds' in problem:
            nodes_loads = load_cond
            # nodes_loads = problem['load_conds'][0]
            ax = plt.gca()  # Get the current axis
            # nodes_loads = problem['nodes_loads']
            for I, (load_x, load_y) in enumerate(nodes_loads):
                # print('LOAD:', I, load_x, load_y)
                if load_x != 0:
                    arrow_start = [nodal_locations[I][0], nodal_locations[I][1]]
                    if load_x > 0:
                        arrow_end = [arrow_start[0] + (x_range / 10), arrow_start[1]]
                    else:
                        arrow_end = [arrow_start[0] - (x_range / 10), arrow_start[1]]
                    ax.arrow(arrow_start[0], arrow_start[1], arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1],
                             head_width=0.2, head_length=0.2, width=0.05,
                             fc='green', ec='green', zorder=2)
                    plt.text(arrow_end[0], arrow_end[1] + (y_range / 100), f'{load_x:.2f} N', fontsize=10,
                             ha='right')  # Annotate nodes with their index

                if load_y != 0:
                    arrow_start = [nodal_locations[I][0], nodal_locations[I][1]]
                    if load_y > 0:
                        arrow_end = [arrow_start[0], arrow_start[1] + (y_range / 10)]
                    else:
                        arrow_end = [arrow_start[0], arrow_start[1] - (y_range / 10)]
                    ax.arrow(arrow_start[0], arrow_start[1], arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1],
                             head_width=0.2, head_length=0.2, width=0.05,
                             fc='green', ec='green', zorder=2)
                    plt.text(arrow_end[0], arrow_end[1] - (x_range / 100), f'{load_y:.2f} N', fontsize=10,
                             ha='right')  # Annotate nodes with their index

        # Adding optional text
        if optional_text is not None:
            plt.figtext(
                0.5 * i, 0.05,
                optional_text,
                ha='center', va='bottom', fontsize=10,
                bbox=dict(facecolor='red', alpha=0.5)
            )
            plt.subplots_adjust(bottom=0.3)

            # plt.text(text_position[0], text_position[1], optional_text, fontsize=12, ha='center',
            #          transform=plt.gca().transAxes)

        # Setting the plot labels and title
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(f'Load Condition {l_idx}')
        plt.grid(True)
        plt.axis('equal')  # Ensure the aspect ratio is equal to avoid distortion


        # Calc text pos
        pos = ax.get_position()  # Get the position of the subplot
        # x_center = (pos.x0 + pos.x1) / 2  # Calculate the center x position
        x_center = pos.x0
        # x_center = 0.05 + (0.5 * l_idx)

    # Design Text
    design_metrics = get_design_metrics(problem, design_rep)
    # wrapped_text = textwrap.fill(design_metrics, width=50)
    wrapped_lines = []
    for paragraph in design_metrics.split('\n'):
        wrapped_lines.extend(textwrap.wrap(paragraph, 50))
        # wrapped_lines.append('')  # Add a newline after each paragraph
    wrapped_text = '\n'.join(wrapped_lines)  # Remove the last empty line
    plt.figtext(
        0.05, y_center,
        wrapped_text,
        ha='left', va='bottom', fontsize=9,
        bbox=dict(facecolor='grey', alpha=0.5),
    )
    plt.tight_layout()

    # Save the plot
    if base_dir is None:
        base_dir = config.plots_dir

    if f_name is None:
        num_files = len([name for name in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, name))])
        f_name = f'truss_{num_files}.png'
    plt.savefig(os.path.join(base_dir, f_name))
    plt.close('all')

def viz_datapoint(problem, sample, f_name=None):
    # Plotting the nodes
    bit_list1, bit_str1, node_idx_pairs1, node_coords1 = convert(problem, sample['design_edit'])
    bit_list2, bit_str2, node_idx_pairs2, node_coords2 = convert(problem, sample['design_fixed'])

    # --- Plotting ---
    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure(figsize=(12, 10))  # default [6.4, 4.8], W x H  9x6, 12x8
    # fig.suptitle('Design Edit Model', fontsize=16)

    # Original design
    plt.subplot(gs[0, 0])
    # Plotting the nodes
    nodal_locations = problem['nodes']
    for i, (x, y) in enumerate(nodal_locations):
        plt.plot(x, y, 'bo')  # Plot nodes as blue circles
        plt.text(x, y, f'{i}', fontsize=12, ha='right')  # Annotate nodes with their index

    # Plotting the truss members
    for (start, end) in node_idx_pairs1:
        x_coords = [nodal_locations[start][0], nodal_locations[end][0]]
        y_coords = [nodal_locations[start][1], nodal_locations[end][1]]
        plt.plot(x_coords, y_coords, 'r-')  # Plot truss members as red lines

    # Setting the plot labels and title
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Base Design')
    plt.grid(True)
    plt.axis('equal')  # Ensure the aspect ratio is equal to avoid distortion

    # Fixed design
    plt.subplot(gs[0, 1])
    # Plotting the nodes
    nodal_locations = problem['nodes']
    for i, (x, y) in enumerate(nodal_locations):
        plt.plot(x, y, 'bo')  # Plot nodes as blue circles
        plt.text(x, y, f'{i}', fontsize=12, ha='right')  # Annotate nodes with their index

    # Plotting the truss members
    for (start, end) in node_idx_pairs2:
        x_coords = [nodal_locations[start][0], nodal_locations[end][0]]
        y_coords = [nodal_locations[start][1], nodal_locations[end][1]]
        plt.plot(x_coords, y_coords, 'r-')  # Plot truss members as red lines

    # Setting the plot labels and title
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Updated Design')
    plt.grid(True)
    plt.axis('equal')  # Ensure the aspect ratio is equal to avoid distortion

    # Design Edit Text
    before_edit_text = get_design_metrics(problem, sample['design_edit'])
    plt.figtext(
        0.12, 0.25,
        before_edit_text,
        ha='left', va='bottom', fontsize=10,
        bbox=dict(facecolor='grey', alpha=0.5),
    )

    # Design Fixed Text
    # plt.subplot(gs[1, 1])
    after_edit_text = get_design_metrics(problem, sample['design_fixed'])
    plt.figtext(
        0.55, 0.25,
        after_edit_text,
        ha='left', va='bottom', fontsize=10,
        bbox=dict(facecolor='grey', alpha=0.5)
    )

    # Feature
    if sample['feature'] is not None:
        wrapped_text = textwrap.fill(sample['feature'], width=500)
        plt.figtext(
            0.5, 0.95,
            wrapped_text,
            ha='center', va='bottom', fontsize=13,
            bbox=dict(facecolor='yellow', alpha=0.5)
        )

    # Add a text box outside the plot
    if sample['info'] is not None:
        # wrapped_text = textwrap.fill(sample['info'], width=150)
        plt.figtext(
            0.05, 0.05,
            sample['info'],
            ha='left', va='bottom', fontsize=10,
            bbox=dict(facecolor='grey', alpha=0.5)
        )

        # plt.subplots_adjust(bottom=0.0)

    # Save the plot
    if f_name is None:
        num_files = len(
            [name for name in os.listdir(config.plots_dir) if os.path.isfile(os.path.join(config.plots_dir, name))])
        f_name = f'truss_{num_files}.png'
    plt.savefig(os.path.join(config.plots_features_dir, f_name))
    plt.close('all')


def get_design_metrics(problem, design_rep):
    from combench.models.truss import eval_volfrac, eval_stiffness

    bit_list, bit_str, node_idx_pairs, node_coords = convert(problem, design_rep)
    # Metrics
    # - Number of truss members
    # - Number of non-overlapping truss members
    # - Used Nodes
    # - Truss Angles
    num_members = sum(bit_list)
    no_overlaps = remove_overlapping_members(problem, design_rep)
    num_members_no_overlaps = sum(no_overlaps)
    used_nodes = get_used_nodes(problem, design_rep)
    design_angle = get_design_angle(problem, design_rep)
    vol_frac = eval_volfrac(problem, design_rep, normalize=True)
    stiff, extra_info = eval_stiffness(problem, design_rep, normalize=False, verbose=True)

    metrics = [
        f'Truss Members: {num_members}',
        f'Non-Overlapping Truss Members: {num_members_no_overlaps}',
        f'Truss Nodes: {used_nodes}',
        f'Truss Angle: {design_angle:.2f} deg',
        '-----',
        f'Youngs Modulus: {problem["youngs_modulus"]:.2e} Pa (N/m^2)',
        f'Member Radii: {problem["member_radii"]} m',
        f'Volume Fraction: {vol_frac:.3f}',
    ]

    stiffness_metrics = []
    for idx, s in enumerate(stiff):
        load_cond_metrics = ['-----']
        load_cond_metrics.append(f'Load Condition {idx + 1}')
        load_cond_metrics.append(f'Stiffness: {s:.2e} N/m')
        if len(extra_info) != 0:
            info_dict = extra_info[idx]
            for key, val in info_dict.items():
                load_cond_metrics.append(f'{key}: {val}')
            stiffness_metrics.extend(load_cond_metrics)
    metrics.extend(stiffness_metrics)
    metrics_text = '\n'.join(metrics)


    return metrics_text

# ------------------------------
# Evaluation
# ------------------------------
# from truss.model.vol.c_geometry import vox_space
# from truss.model.stiffness_old.Stiffness import Stiffness


# def eval_volfrac(problem, design_rep):
#     bit_list, bit_str, node_idx_pairs, node_coords = convert(problem, design_rep)
#     vol_frac = vox_space(problem, node_idx_pairs, resolution=100)
#     return vol_frac


# def eval_stiffness(problem, design_rep):
#     return Stiffness().evaluate(problem, design_rep)
#


if __name__ == '__main__':
    problem = {
        'nodes': [  # 4x4
            (0, 0), (0, 1), (0, 2), (0, 3),
            (1, 0), (1, 1), (1, 2), (1, 3),
            (2, 0), (2, 1), (2, 2), (2, 3),
            (3, 0), (3, 1), (3, 2), (3, 3),
        ],
        'member_radii': 0.1,  # This can either be a list or a single value
        'youngs_modulus': 1.8162e6,
    }

    d1_sample = random_sample_1(problem)
    d2_sample = random_sample_2(problem)
    d3_sample = random_sample_3(problem)

    viz(problem, d1_sample, 'd1_sample.png')
    viz(problem, d2_sample, 'd2_sample.png')
    viz(problem, d3_sample, 'd3_sample.png')

    exit(0)

    design = [0 for _ in range(120)]
    design[0] = 1
    design[1] = 1
    design[22] = 1

    bit_angles = get_bit_angles(problem)
    print(len(bit_angles), bit_angles)

    angle = get_design_angle(problem, design)
    print(angle)
    exit(0)

    # design = '000000000010000000000000000000000000'
    # design = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # design = [(1, 4)]
    # design = [[(0, 1), (1, 1)]]
    # convert(problem, design)

    sample = random_sample_3(problem)
    print(sample)
