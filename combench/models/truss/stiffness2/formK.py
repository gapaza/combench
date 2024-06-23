import numpy as np


def Kel(node1, node2):
    '''Kel(node1,node2) returns the diagonal and off-diagonal element stiffness_old matrices based upon
    initial angle of a beam element and its length the full element stiffness_old is
    K_el = np.block([[Ke1,Ke2],[Ke2,Ke1]])

    Out: [Ke1 Ke2]
         [Ke2 Ke1]
    arguments:
    ----------
    node1: is the 1st node number and coordinates from the nodes array
    node2: is the 2nd node number and coordinates from the nodes array
    outputs:
    --------
    Ke1 : the diagonal matrix of the element stiffness_old
    Ke2 : the off-diagonal matrix of the element stiffness_old
    '''
    a = np.arctan2(node2[2] - node1[2], node2[1] - node1[1])
    l = np.sqrt((node2[2] - node1[2]) ** 2 + (node2[1] - node1[1]) ** 2)
    Ke1 = 1 / l * np.array([[np.cos(a) ** 2, np.cos(a) * np.sin(a)], [np.cos(a) * np.sin(a), np.sin(a) ** 2]])
    Ke2 = 1 / l * np.array([[-np.cos(a) ** 2, -np.cos(a) * np.sin(a)], [-np.cos(a) * np.sin(a), -np.sin(a) ** 2]])
    return Ke1, Ke2

def fullK(node_coords, node_connectivity, avar, E):
    nodes = node_coords
    elems = node_connectivity

    K = np.zeros((len(nodes) * 2, len(nodes) * 2))
    for e in elems:
        ni = nodes[e[1] - 1]
        nj = nodes[e[2] - 1]

        Ke1, Ke2 = Kel(ni, nj)

        Ke1 = Ke1 * avar[e[0] - 1] * E
        Ke2 = Ke2 * avar[e[0] - 1] * E

        # --> assemble K <--
        i1 = int(ni[0]) * 2 - 2
        i2 = int(ni[0]) * 2
        j1 = int(nj[0]) * 2 - 2
        j2 = int(nj[0]) * 2

        K[i1:i2, i1:i2] += Ke1
        K[j1:j2, j1:j2] += Ke1
        K[i1:i2, j1:j2] += Ke2
        K[j1:j2, i1:i2] += Ke2

    # print(K, K.shape)
    return K




