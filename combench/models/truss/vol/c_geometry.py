import numpy as np
from scipy.spatial import KDTree, cKDTree
import time
from copy import deepcopy
import config
import math


def vox_space_trivial(problem, connectivity_array, resolution=100):
    nodes = deepcopy(problem['nodes'])
    nodes = np.array(nodes).tolist()
    member_radii = problem['member_radii']
    radius = member_radii
    member_cs_area = (np.pi * radius) ** 2
    # Calculate volume of each member
    member_vols = []
    for ca in connectivity_array:
        p1 = nodes[ca[0]]
        p2 = nodes[ca[1]]
        dist = np.linalg.norm(np.array(p1) - np.array(p2))
        member_vol = dist * member_cs_area
        member_vols.append(member_vol)
    total_vol = sum(member_vols)
    return total_vol





def vox_space(problem, connectivity_array, resolution=100):
    nodes = deepcopy(problem['nodes'])
    nodes = np.array(nodes).tolist()

    x_coords = [node[0] for node in nodes]
    y_coords = [node[1] for node in nodes]
    x_range = max(x_coords) - min(x_coords)
    y_range = max(y_coords) - min(y_coords)


    member_radii = problem['member_radii']
    radius = member_radii

    # Add depth as third dimension
    for idx in range(len(nodes)):
        nodes[idx].append(0)
    nodal_coords = np.array(nodes)

    # 1. Create bounding box
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

    # 2. Create voxel grid
    nodes_used = set()
    for ca in connectivity_array:
        nodes_used.add(ca[0])
        nodes_used.add(ca[1])
    nodes_used = list(nodes_used)
    if len(nodes_used) == 0:
        return 0.0
    node_x_vals = [nodal_coords[node][0] for node in nodes_used]
    node_y_vals = [nodal_coords[node][1] for node in nodes_used]
    x_min, x_max = min(node_x_vals) - radius, max(node_x_vals) + radius
    y_min, y_max = min(node_y_vals) - radius, max(node_y_vals) + radius

    # --- Voxel Grid
    # w_res = min(int(width / (radius*0.5)), resolution)
    # h_res = min(int(height / (radius*0.5)), resolution)
    # d_res = min(25, resolution)

    w_res = resolution
    h_res = resolution
    d_res = resolution


    voxel_width = width / w_res
    voxel_height = height / h_res
    voxel_depth = depth / d_res


    # print('Voxel Sizes:', w_res, h_res, d_res)
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
    # tree = KDTree(voxel_centers)
    tree = cKDTree(voxel_centers)
    intersected_voxels = set()

    all_points = []
    for element in connectivity_array:
        start_node = np.array(nodal_coords[element[0]])
        end_node = np.array(nodal_coords[element[1]])
        direction = end_node - start_node
        length = np.linalg.norm(direction)
        direction = direction / length

        step_len = radius / 10
        num_steps = math.floor(length / step_len)
        # print('NUM STEPS:', num_steps)
        num_steps = min(num_steps, 100)
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







