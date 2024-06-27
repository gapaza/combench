import numpy as np


def formK(NC, CA, Avar, E):
    """
    Function to form the global structural stiffness matrix.

    Parameters:
    NC (np.array): Nodal coordinates matrix. Each row represents a node [x, y].
    CA (np.array): Connectivity array. Each row represents an element [node1_index, node2_index].
    Avar (np.array): Cross-sectional areas of each element.
    E (float): Young's modulus.

    Returns:
    np.array: Global stiffness matrix.
    """
    num_nodes = NC.shape[0]
    num_elements = CA.shape[0]
    K = np.zeros((2 * num_nodes, 2 * num_nodes))

    for i in range(num_elements):
        node1, node2 = CA[i] - 1  # Convert from 1-based to 0-based indexing
        x1, y1 = NC[node1]
        x2, y2 = NC[node2]

        L = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if L == 0:
            # print(f"Element {i} has zero length. Skipping...")
            continue

        c = (x2 - x1) / L
        s = (y2 - y1) / L
        c2 = c ** 2
        s2 = s ** 2
        cs = c * s

        k_temp = np.array([
            [c2, cs, -c2, -cs],
            [cs, s2, -cs, -s2],
            [-c2, -cs, c2, cs],
            [-cs, -s2, cs, s2]
        ])

        ke = (Avar[i] * E / L) * k_temp

        global_dof = np.array([
            2 * node1, 2 * node1 + 1,
            2 * node2, 2 * node2 + 1
        ])

        for lr in range(4):
            gr = global_dof[lr]
            for lc in range(4):
                gc = global_dof[lc]
                K[gr, gc] += ke[lr, lc]

    return K



