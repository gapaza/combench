import numpy as np
from copy import deepcopy
import math
import matplotlib.pyplot as plt


def compare_points(p1, p2):
    return p1[0] == p2[0] and p1[1] == p2[1]




def do_lines_intersect(p1, q1, p2, q2):
    def on_segment(p, q, r):
        if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
            return True
        return False

    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0  # collinear
        return 1 if val > 0 else 2  # clock or counterclock wise

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and on_segment(p1, p2, q1):
        return True
    if o2 == 0 and on_segment(p1, q2, q1):
        return True
    if o3 == 0 and on_segment(p2, p1, q2):
        return True
    if o4 == 0 and on_segment(p2, q1, q2):
        return True

    return False


def calculate_slope_length(x1, y1, x2, y2):
    if x2 - x1 == 0:
        slope = float('inf')  # vertical line
    else:
        slope = (y2 - y1) / (x2 - x1)
    length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return slope, length

def line_intersection(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    p1_slope, p1_length = calculate_slope_length(x1, y1, x2, y2)
    p2_slope, p2_length = calculate_slope_length(x3, y3, x4, y4)
    if p1_slope == p2_slope:
        return None



    # Calculate the determinants
    det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if det == 0:
        return None  # Lines are parallel

    det1 = (x1 * y2 - y1 * x2)
    det2 = (x3 * y4 - y3 * x4)

    x = (det1 * (x3 - x4) - (x1 - x2) * det2) / det
    y = (det1 * (y3 - y4) - (y1 - y2) * det2) / det

    # Is the intersection point a start or end point for either line?
    if (x, y) == (x1, y1) or (x, y) == (x2, y2) or (x, y) == (x3, y3) or (x, y) == (x4, y4):
        return None

    return x, y


def visualize_graph(nodes, edges):
    """
    Visualize a graph given nodes and edges.

    Parameters:
    nodes (list of tuple): List of (x, y) coordinates of the nodes.
    edges (list of tuple): List of pairs (i, j) where i and j are indices of nodes being connected.
    """
    # Create a figure and a set of subplots
    fig, ax = plt.subplots()

    # Plot the nodes
    for i, (x, y) in enumerate(nodes):
        ax.plot(x, y, 'o', label=f'Node {i}')
        ax.text(x, y, f'{i}', fontsize=12, ha='right')

    # Plot the edges
    for i, j in edges:
        x_values = [nodes[i][0], nodes[j][0]]
        y_values = [nodes[i][1], nodes[j][1]]
        ax.plot(x_values, y_values, 'k-')

    # Set labels and title
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('Graph Visualization')

    # Show legend
    # ax.legend()

    # Show the plot
    plt.show()

def check_connection(conn, conn_list):
    for c in conn_list:
        if conn[0] in c and conn[1] in c:
            return True
    return False

def add_intersection_nodes(nodes, connection_list):
    # nodes: list of node coordinates
    # connection_list: list of node pairs that are connected

    # visualize_graph(nodes, connection_list)

    # print('Initial Connections:', connection_list)

    new_nodes = deepcopy(nodes)
    new_connections = deepcopy(connection_list)

    to_remove = []
    to_add = []

    # Iterate over all pairs of connections
    total_intersections = 0
    line_intersections = {}
    found = False
    for i, (c1_start_node_idx, c1_end_node_idx) in enumerate(connection_list):
        for j, (c2_start_node_idx, c2_end_node_idx) in enumerate(connection_list):
            if i >= j:
                continue
            c1_start_node, c1_end_node = nodes[c1_start_node_idx], nodes[c1_end_node_idx]
            c2_start_node, c2_end_node = nodes[c2_start_node_idx], nodes[c2_end_node_idx]

            # Check if connections share a common node
            c2_nodes = [c2_start_node_idx, c2_end_node_idx]
            if c1_start_node_idx in c2_nodes or c1_end_node_idx in c2_nodes:
                continue

            # Check if lines intersect
            if do_lines_intersect(c1_start_node, c1_end_node, c2_start_node, c2_end_node) is True:

                # print("Intersection found", c1_start_node, c1_end_node, ' --> ', c2_start_node, c2_end_node)
                # print("Intersection found", c1_start_node_idx, c1_end_node_idx, ' --> ', c2_start_node_idx, c2_end_node_idx)
                intersection = line_intersection(c1_start_node, c1_end_node, c2_start_node, c2_end_node)
                # print("Intersection at:", intersection)

                # Check if lines are parallel
                if intersection is None:
                    # print("\n\nLines are parallel", c1_start_node_idx, c1_end_node_idx, ' --> ', c2_start_node_idx, c2_end_node_idx)
                    continue



                line_pair = (c1_start_node_idx, c1_end_node_idx)
                if line_pair not in line_intersections:
                    line_intersections[line_pair] = []

                intersection_obj = {
                    'intersects': (c2_start_node_idx, c2_end_node_idx),
                    'point': intersection
                }
                line_intersections[line_pair].append(intersection_obj)

                if intersection:
                    total_intersections += 1
                    if intersection not in new_nodes:
                        new_nodes.append(intersection)
                        intersection_idx = len(new_nodes) - 1
                    else:
                        intersection_idx = new_nodes.index(intersection)

                    c1_sub_conn1 = [c1_start_node_idx, intersection_idx]
                    c1_sub_conn2 = [intersection_idx, c1_end_node_idx]
                    if check_connection(c1_sub_conn1, new_connections) is False:
                        to_add.append(c1_sub_conn1)
                    if check_connection(c1_sub_conn2, new_connections) is False:
                        to_add.append(c1_sub_conn2)

                    c2_sub_conn1 = [c2_start_node_idx, intersection_idx]
                    c2_sub_conn2 = [intersection_idx, c2_end_node_idx]
                    if check_connection(c2_sub_conn1, new_connections) is False:
                        to_add.append(c2_sub_conn1)
                    if check_connection(c2_sub_conn2, new_connections) is False:
                        to_add.append(c2_sub_conn2)

                    if i not in to_remove:
                        to_remove.append(i)
                    if j not in to_remove:
                        to_remove.append(j)

                    found = True
                    break
        if found is True:
            break





    remove_conns = [connection_list[i] for i in to_remove]
    # print('Conns to remove:', remove_conns)

    # for conn in to_remove:
    #     if conn in new_connections:
    #         new_connections.remove(conn)
    # Iterate over new_connections backwards, and remove indices in to_remove
    # sort to_remove in descending order
    to_remove.sort(reverse=True)
    for i in to_remove:
        del new_connections[i]
    new_connections.extend(to_add)

    # unique_new_connections = []
    # for conn in new_connections:
    #     if check_connection(conn, unique_new_connections) is False:
    #         unique_new_connections.append(conn)
    # new_connections = unique_new_connections


    # print("Total intersections found:", total_intersections)
    #
    # print('Num Connections:', len(new_connections))
    return new_nodes, new_connections, found


if __name__ == '__main__':
    NC = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Nodal coordinates
    CA = np.array([[1, 2], [2, 3], [3, 4], [4, 1], [1, 3], [2, 4]])  # Connectivity array
    CA = CA - 1
    nodes = NC.tolist()

    connections = CA.tolist()
    print(connections)


    new_nodes, new_connections = add_intersection_nodes(nodes, connections)
    print("Updated Nodes:", new_nodes)
    print("Updated Connections:", np.array(new_connections)+1)
    print("Number of Nodes:", len(new_nodes))
