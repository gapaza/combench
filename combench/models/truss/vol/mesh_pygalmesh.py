import pygalmesh
import numpy as np




if __name__ == '__main__':
    # Problem Parameters
    radius = 0.2
    nodes = [[0.0, 0.0], [0.0, 1.3333333333333333], [0.0, 2.6666666666666665], [0.0, 4.0], [1.3333333333333333, 0.0],
             [1.3333333333333333, 1.3333333333333333], [1.3333333333333333, 2.6666666666666665],
             [1.3333333333333333, 4.0], [2.6666666666666665, 0.0], [2.6666666666666665, 1.3333333333333333],
             [2.6666666666666665, 2.6666666666666665], [2.6666666666666665, 4.0], [4.0, 0.0], [4.0, 1.3333333333333333],
             [4.0, 2.6666666666666665], [4.0, 4.0]]
    node_edges = [(0, 5), (0, 7), (0, 8), (0, 9), (0, 11), (1, 2), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 10),
                  (1, 12), (1, 14), (1, 15), (2, 4), (2, 6), (2, 7), (2, 9), (2, 11), (2, 12), (2, 13), (2, 15), (4, 5),
                  (4, 10), (4, 13), (4, 14), (5, 6), (5, 8), (5, 9), (5, 10), (5, 11), (5, 12), (6, 8), (6, 9), (6, 10),
                  (6, 11), (6, 15), (7, 10), (7, 12), (7, 14), (8, 9), (8, 12), (8, 13), (8, 14), (9, 10), (9, 12),
                  (9, 13), (9, 15), (10, 11), (11, 14), (11, 15), (14, 15)]

    # Find bounding box volume
    x_coords = [node[0] for node in nodes]
    y_coords = [node[1] for node in nodes]
    x_range = max(x_coords) - min(x_coords)
    y_range = max(y_coords) - min(y_coords)
    x_1 = 0 - radius
    x_2 = x_range + radius
    y_1 = 0 - radius
    y_2 = y_range + radius
    z_1 = 0 - radius
    z_2 = 0 + radius
    width = abs(x_2 - x_1)
    height = abs(y_2 - y_1)
    depth = abs(z_2 - z_1)
    bounding_box_volume = depth * width * height

    global_mesh = None
    for edge in node_edges:
        node1 = nodes[edge[0]]
        node2 = nodes[edge[1]]
        x1, y1 = node1
        x2, y2 = node2
        z1, z2 = 0.0, 0.0
        p1 = np.array([x1, y1, z1])
        p2 = np.array([x2, y2, z2])

        cylinder = pygalmesh.Cylinder()










