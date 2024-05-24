import numpy as np
from scipy.spatial import KDTree
from combench.models.truss.stiffness.generateNC import generateNC
import time

import config

def voxelize_space(radius, sidelen, nodal_coords, connectivity_array, resolution=100):

    # Add depth as third dimension
    nodal_coords = nodal_coords.tolist()
    for idx in range(len(nodal_coords)):
        nodal_coords[idx].append(0)
    nodal_coords = np.array(nodal_coords)

    # 1. Create bounding box
    x_1 = 0 - radius
    x_2 = sidelen + radius
    y_1 = 0 - radius
    y_2 = sidelen + radius
    z_1 = 0 - radius
    z_2 = 0 + radius
    width = abs(x_2 - x_1)
    height = abs(y_2 - y_1)
    depth = abs(z_2 - z_1)
    bounding_box_volume = depth * width * height


    # 2. Create voxel grid
    nodes_used = set()
    for ca in connectivity_array:
        nodes_used.add(ca[0]-1)
        nodes_used.add(ca[1]-1)
    nodes_used = list(nodes_used)
    node_x_vals = [nodal_coords[node][0] for node in nodes_used]
    node_y_vals = [nodal_coords[node][1] for node in nodes_used]
    x_min, x_max = min(node_x_vals) - radius, max(node_x_vals) + radius
    y_min, y_max = min(node_y_vals) - radius, max(node_y_vals) + radius

    # voxel_size = depth / 10
    # voxel_size = width / resolution
    voxel_width = width / resolution
    voxel_height = height / resolution
    voxel_depth = depth / resolution
    # print('Voxel Sizes:', voxel_width, voxel_height, voxel_depth)
    x, y, z = np.meshgrid(
        # np.arange(x_1, x_2, voxel_width),
        # np.arange(y_1, y_2, voxel_height),
        # np.arange(z_1, z_2, voxel_depth),
        np.arange(x_min, x_max, voxel_width),
        np.arange(y_min, y_max, voxel_height),
        np.arange(z_1, z_2, voxel_depth),
        indexing='ij'
    )
    voxel_centers = np.vstack((x.ravel(), y.ravel(), z.ravel())).T
    # print('Number of Voxels:', len(voxel_centers))

    # Use a KDTree for efficient distance queries
    tree = KDTree(voxel_centers)
    intersected_voxels = set()

    all_points = []
    for element in connectivity_array:
        start_node = np.array(nodal_coords[element[0]-1])
        end_node = np.array(nodal_coords[element[1]-1])
        direction = end_node - start_node
        length = np.linalg.norm(direction)
        direction = direction / length

        # step_len = radius / 10
        # num_steps = math.floor(length / step_len)
        num_steps = 100
        step_len = length / num_steps
        for i in range(num_steps):
            point = start_node + i * step_len * direction
            all_points.append(point)
            # idx = tree.query_ball_point(point, radius, workers=1)
            # intersected_voxels.update(idx)

    all_points = np.array(all_points)
    idx = tree.query_ball_point(all_points, radius, workers=-1)
    for i in idx:
        intersected_voxels.update(i)

    # voxel_volume = voxel_size ** 3
    voxel_volume = voxel_width * voxel_height * voxel_depth
    truss_volume = len(intersected_voxels) * voxel_volume
    volume_fraction = truss_volume / bounding_box_volume
    # print('Truss Volume:', truss_volume)  # 0.015307337294603601
    return volume_fraction











if __name__ == '__main__':
    sidenum = 3
    num_vars = config.sidenum_nvar_map[sidenum]
    radius = 0.1
    sidelen = 1  # length of the side of the truss (not a single truss)


    # NC = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Nodal coordinates
    # CA = [[1, 2], [2, 3], [3, 4], [4, 1], [1, 3], [2, 4]]  # Connectivity array

    NC = generateNC(sidelen, sidenum)
    print(NC)

    CA = np.array([
        [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9],
        [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [2, 9],
        [3, 4], [3, 5], [3, 6], [3, 7], [3, 8], [3, 9],
        [4, 5], [4, 6], [4, 7], [4, 8], [4, 9],
        [5, 6], [5, 7], [5, 8], [5, 9],
        [6, 7], [6, 8], [6, 9],
        [7, 8], [7, 9],
        [8, 9],
    ])  # Connectivity array


    curr_time = time.time()
    volume_fraction = voxelize_space(radius, sidelen, NC, CA, resolution=100)
    print('Volume Fraction:', volume_fraction)
    print('Time taken:', time.time() - curr_time)

    # 0.015307337294603601

    # print(f"Volume Fraction of the Truss Structure: {volume_fraction:.6f}")









