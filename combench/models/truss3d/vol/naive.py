import numpy as np
from combench.models.truss3d import representation as rep



def eval_volfrac(problem, design_rep):
    bit_list, bit_str, node_idx_pairs, node_coords = rep.convert(problem, design_rep)

    nodes = problem['nodes']
    radius = problem['member_radii']
    member_cs_area = (np.pi * radius) ** 2

    # Accumulate volume of each member
    member_vols = []
    for ca in node_idx_pairs:
        p1 = nodes[ca[0]]
        p2 = nodes[ca[1]]
        x_dist = p2[0] - p1[0]
        y_dist = p2[1] - p1[1]
        z_dist = p2[2] - p1[2]
        dist = np.sqrt(x_dist**2 + y_dist**2 + z_dist**2)
        member_vol = dist * member_cs_area
        member_vols.append(member_vol)
    total_vol = sum(member_vols)
    return total_vol
