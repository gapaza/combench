import numpy as np


def modifyAreas(Avar, CA, NC, sidenum):
    # Identify edge nodes
    edgenodes = np.concatenate([
        np.arange(1, sidenum + 1),
        np.arange(2 * sidenum, sidenum ** 2 - sidenum + 1, sidenum),
        np.arange(sidenum + 1, sidenum ** 2 - (2 * sidenum) + 2, sidenum),
        np.arange((sidenum ** 2) - sidenum + 1, sidenum ** 2 + 1)
    ])

    # print('Edge nodes:', edgenodes)

    # Identify members connecting solely to edge nodes
    edgeconn1 = np.isin(CA[:, 0], edgenodes)
    edgeconn2 = np.isin(CA[:, 1], edgenodes)
    edgeconnectors = edgeconn1 & edgeconn2

    # print('Edgeconn1:', edgeconn1)
    # print('Edgeconn2:', edgeconn2)
    # print('Edgeconnectors:', edgeconnectors)

    # Isolate edge members based on angle
    CAedgenodes = CA * edgeconnectors[:, np.newaxis]
    CAedgenodes = CAedgenodes[np.any(CAedgenodes, axis=1)]
    x1 = NC[CAedgenodes[:, 0] - 1, 0]
    x2 = NC[CAedgenodes[:, 1] - 1, 0]
    y1 = NC[CAedgenodes[:, 0] - 1, 1]
    y2 = NC[CAedgenodes[:, 1] - 1, 1]
    L = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    angles = np.rad2deg(np.abs(np.arccos((x2 - x1) / L)))
    CAedgy = []
    for i in range(len(CAedgenodes)):
        if angles[i] == 0 or angles[i] == 90:
            CAedgy.append(CAedgenodes[i])
    CAedgy = np.array(CAedgy)

    # Find and modify areas belonging to edge members
    if CAedgy.size > 0:
        # edgemembers = np.isin(CA, CAedgy).all(axis=1)
        # print('CA:', CA)
        # print('CAedgy:', CAedgy)
        # print('Edgemembers:', edgemembers)

        edge_members = []
        for edge in CA:
            if search_pairs(edge, CAedgy):
                edge_members.append(True)
            else:
                edge_members.append(False)


        selectAreas = (Avar * edge_members)
        k = np.where(selectAreas)[0]
        Avar[k] = Avar[k] / 2

    return Avar


def search_pairs(pair, search_list):
    for s_pair in search_list:
        if pair[0] in s_pair and pair[1] in s_pair:
            return True
    return False








if __name__ == '__main__':
    CA = np.array([
        [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9],
        [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [2, 9],
        [3, 4], [3, 5], [3, 6], [3, 7], [3, 8], [3, 9],
        [4, 5], [4, 6], [4, 7], [4, 8], [4, 9],
        [5, 6], [5, 7], [5, 8], [5, 9],
        [6, 7], [6, 8], [6, 9],
        [7, 8], [7, 9],
        [8, 9]
    ])
    NC = np.array([
        [0.0, 0.0],
        [0.0, 0.005],
        [0.0, 0.01],
        [0.005, 0.0],
        [0.005, 0.005],
        [0.005, 0.01],
        [0.01, 0.0],
        [0.01, 0.005],
        [0.01, 0.01]
    ])

    rvar = (250e-6) * np.ones(CA.shape[0])
    Avar = np.pi * (rvar ** 2)

    print('Avar IN:', Avar)


    sidenum = 3  # Example side number

    # Modify areas
    Avar_modified = modifyAreas(Avar, CA, NC, sidenum)
    print('Avar OUT:', Avar_modified)











