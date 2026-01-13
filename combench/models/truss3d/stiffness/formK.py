import numpy as np


def formK(NC, CA, Avar, E):
    """
    Function to form the global structural stiffness matrix for a 3D truss.

    Parameters:
    NC (np.array): Nodal coordinates matrix. Each row represents a node [x, y, z].
    CA (np.array): Connectivity array. Each row represents an element [node1_index, node2_index].
                   Note: Assumes 0-based for consistency.
    Avar (np.array): Cross-sectional areas of each element.
    E (float): Young's modulus.

    Returns:
    np.array: Global stiffness matrix of size (3*num_nodes, 3*num_nodes).
    """
    num_nodes = NC.shape[0]
    num_elements = CA.shape[0]

    # Each node has 3 degrees of freedom (x, y, z)
    K = np.zeros((3 * num_nodes, 3 * num_nodes))

    for i in range(num_elements):
        # Convert from 1-based to 0-based indexing
        node1, node2 = CA[i]

        x1, y1, z1 = NC[node1]
        x2, y2, z2 = NC[node2]

        # Calculate Length
        L = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

        if L <= 1e-12:  # Numerical stability check
            continue

        # Direction Cosines
        cx = (x2 - x1) / L
        cy = (y2 - y1) / L
        cz = (z2 - z1) / L

        # Local stiffness matrix components
        # The 6x6 matrix for a 3D truss element is (AE/L) * [gamma^T * gamma]
        # where gamma = [cx, cy, cz, -cx, -cy, -cz]
        gamma = np.array([cx, cy, cz, -cx, -cy, -cz])

        # Element stiffness matrix using outer product: shape (6, 6)
        ke = (Avar[i] * E / L) * np.outer(gamma, gamma)

        # Global Degree of Freedom mapping
        # Node i -> indices [3i, 3i+1, 3i+2]
        dofs = np.array([
            3 * node1, 3 * node1 + 1, 3 * node1 + 2,
            3 * node2, 3 * node2 + 1, 3 * node2 + 2
        ])

        # Assembly into Global Matrix
        # Using np.ix_ allows for vectorized assembly of the 6x6 block
        K[np.ix_(dofs, dofs)] += ke

    return K