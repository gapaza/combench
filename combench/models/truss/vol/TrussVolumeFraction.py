import math
import config
import numpy as np

from combench.models.truss.TrussFeatures import TrussFeatures, intersect, calculate_angle, estimate_intersection_volume

from combench.models.truss.vol.c_geometry import voxelize_space, generateNC
from combench.models.truss.vol.decompose import add_intersection_nodes, visualize_graph, check_connection
# Calculate bit connection list
bit_connection_list = []

for x in range(config.num_vars):
    design = [0 for _ in range(config.num_vars)]
    design[x] = 1
    truss_features = TrussFeatures(design, config.sidenum, None)
    design_conn_array = truss_features.design_conn_array
    bit_connection_list.append(design_conn_array)

    # print('--> Bit:', x, 'Connections:', design_conn_array)





class TrussVolumeFraction:

    def __init__(self, sidenum, bit_list, side_length=100e-5):
        self.bit_list = bit_list
        self.sidenum = sidenum
        self.sidelen = side_length
        self.member_length = side_length / (sidenum - 1)
        # self.num_members = TrussFeatures.get_member_count(self.sidenum)
        # self.num_repeatable_members = TrussFeatures.get_member_count_repeatable(self.sidenum)

        self.feasibility_constraint_norm = 1
        if self.sidenum == 3:
            self.feasibility_constraint_norm = 94
        elif self.sidenum == 4:
            self.feasibility_constraint_norm = 1304
        elif self.sidenum == 5:
            self.feasibility_constraint_norm = 8942
        elif self.sidenum == 6:
            self.feasibility_constraint_norm = 41397
        # else:
        #     raise ValueError('Invalid sidenum:', self.sidenum)

        self.n = config.sidenum_nvar_map[self.sidenum]
        # self.feasibility_constraint_norm = config.feasibility_constraint_norm_map[self.sidenum]

        self.truss_features = TrussFeatures(bit_list, sidenum, None)
        self.design_conn_array = self.truss_features.design_conn_array

        self.node_positions = {}
        idx = 1
        for x in range(self.sidenum):
            for y in range(self.sidenum):
                self.node_positions[idx] = (x * self.member_length, y * self.member_length)
                idx += 1


    def visualize(self):
        self.truss_features.visualize_design()


    def get_bit_connections(self, bit_idx):
        return bit_connection_list[bit_idx]


    def get_bits_from_connection(self, connection):
        bits = []
        for idx, bit_conns in enumerate(bit_connection_list):
            for bit_conn in bit_conns:
                if connection[0] in bit_conn and connection[1] in bit_conn:
                    bits.append(idx)
        return bits

    def evaluate(self, member_radii=50e-6, side_length=100e-5):
        # sidelen = side_length / (self.sidenum - 1)
        sidelen = side_length
        CA = np.array(self.design_conn_array)
        NC = generateNC(sidelen, self.sidenum)
        volume_fraction = voxelize_space(member_radii, sidelen, NC, CA, resolution=50)
        return volume_fraction, 1.0, [0 for _ in range(config.num_vars)]


    def evaluate_decomp(self, member_radii=50e-6, side_length=100e-5):
        sidelen = side_length
        CA = np.array(self.design_conn_array)
        NC = generateNC(sidelen, self.sidenum)
        NC, CA = add_intersection_nodes(NC.tolist(), CA.tolist())
        volume_fraction = voxelize_space(member_radii, sidelen, NC, CA, resolution=50)
        return volume_fraction, 1.0, [0 for _ in range(config.num_vars)]


    def evaluate2(self, member_radii=50e-6, side_length=100e-5):

        # 1. calculate total volume of truss
        depth = member_radii * 2
        width = side_length * (self.sidenum - 1)
        height = side_length * (self.sidenum - 1)
        total_volume = depth * width * height

        # 2. calculate total volume of truss members
        member_volumes = []
        member_lengths = []
        member_positions = []
        for connection in self.design_conn_array:
            node1, node2 = connection
            pos1, pos2 = self.node_positions[node1], self.node_positions[node2]
            pos1 = [x * side_length for x in pos1]
            pos2 = [x * side_length for x in pos2]
            member_positions.append((pos1, pos2))
            length = math.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
            member_lengths.append(length)
            member_volume = math.pi * member_radii**2 * length
            member_volumes.append(member_volume)




        # print(self.design_conn_array, member_volumes)

        # 3. Account for truss member overlaps at nodes
        node_volumes = []
        node_volume = (4 / 3) * math.pi * member_radii ** 3
        sub_vol = math.pi * member_radii ** 2 * member_radii
        num_nodes = self.sidenum ** 2
        for x in range(1, num_nodes+1):
            connected_members_indices = [i for i, connection in enumerate(self.design_conn_array) if x in connection]
            if len(connected_members_indices) > 0:
                node_volumes.append(node_volume)
            for idx in connected_members_indices:
                member_volumes[idx] -= sub_vol

        # 4. Account for non-node truss member overlaps
        intersect_vols = []
        num_intersections = 0
        bit_interactions = {}
        for member_a_idx in range(len(self.design_conn_array)):
            member_a_n1, member_a_n2 = self.design_conn_array[member_a_idx]
            a_pos_1, a_pos_2 = self.node_positions[member_a_n1], self.node_positions[member_a_n2]

            # Get bit idx of member a
            # print(self.design_conn_array[member_a_idx])
            bits_a = self.get_bits_from_connection(self.design_conn_array[member_a_idx])

            for member_b_idx in range(member_a_idx + 1, len(self.design_conn_array)):

                # Get bit idx of member b
                bits_b = self.get_bits_from_connection(self.design_conn_array[member_b_idx])

                member_b_n1, member_b_n2 = self.design_conn_array[member_b_idx]
                b_pos_1, b_pos_2 = self.node_positions[member_b_n1], self.node_positions[member_b_n2]
                intersects = intersect(a_pos_1, a_pos_2, b_pos_1, b_pos_2)
                if intersects is True:
                    # print('--> INTERSECTS:', self.design_conn_array[member_a_idx], self.design_conn_array[member_b_idx])
                    # print('-- MORE INFO:', a_pos_1, a_pos_2, b_pos_1, b_pos_2)
                    num_intersections += 1
                    angle = calculate_angle(a_pos_1, a_pos_2, b_pos_1, b_pos_2)
                    volume = estimate_intersection_volume(member_radii, angle)
                    intersect_vols.append(volume)
                    # print('Interaction: ', bits_a, bits_b)
                    for a_bit in bits_a:
                        for b_bit in bits_b:
                            if a_bit not in bit_interactions:
                                bit_interactions[a_bit] = []
                            if b_bit not in bit_interactions:
                                bit_interactions[b_bit] = []
                            if b_bit not in bit_interactions[a_bit]:
                                bit_interactions[a_bit].append(b_bit)
                            if a_bit not in bit_interactions[b_bit]:
                                bit_interactions[b_bit].append(a_bit)


        # 5. Account for parallel truss member overlaps
        y_axis_nodes = [[] for _ in range(self.sidenum)]
        x_axis_nodes = [[] for _ in range(self.sidenum)]
        for node in range(1, num_nodes + 1):
            x, y = node % self.sidenum, (node - 1) // self.sidenum
            y_axis_nodes[x].append(node)
            x_axis_nodes[y].append(node)
        y_axis_nodes = sorted(y_axis_nodes, key=lambda x: x[0])
        all_axis_nodes = x_axis_nodes + y_axis_nodes
        truss_members_vol_ignore = []
        axis_vols = []
        for axis_nodes in all_axis_nodes:
            axis_vol = 0
            axis_node_ranges = []
            for idx, connection in enumerate(self.design_conn_array):
                if connection[0] in axis_nodes and connection[1] in axis_nodes:
                    if idx not in truss_members_vol_ignore:
                        truss_members_vol_ignore.append(idx)
                    axis_node_idx_1 = axis_nodes.index(connection[0])
                    axis_node_idx_2 = axis_nodes.index(connection[1])
                    min_axis_node_idx = min(axis_node_idx_1, axis_node_idx_2)
                    max_axis_node_idx = max(axis_node_idx_1, axis_node_idx_2)
                    axis_node_range = [min_axis_node_idx, max_axis_node_idx]
                    if len(axis_node_ranges) == 0:
                        axis_node_ranges.append(axis_node_range)
                    else:
                        new_range = True
                        for axis_node_range in axis_node_ranges:
                            if axis_node_range[0] <= min_axis_node_idx <= axis_node_range[1] or axis_node_range[0] <= max_axis_node_idx <= axis_node_range[1]:
                                if new_range is False:
                                    num_intersections += 1
                                    bits_a = self.get_bits_from_connection(self.design_conn_array[idx])
                                    bits_b = self.get_bits_from_connection(self.design_conn_array[axis_node_range[0]])
                                    for a_bit in bits_a:
                                        for b_bit in bits_b:
                                            if a_bit not in bit_interactions:
                                                bit_interactions[a_bit] = []
                                            if b_bit not in bit_interactions:
                                                bit_interactions[b_bit] = []
                                            if b_bit not in bit_interactions[a_bit]:
                                                bit_interactions[a_bit].append(b_bit)
                                            if a_bit not in bit_interactions[b_bit]:
                                                bit_interactions[b_bit].append(a_bit)
                                new_range = False
                                if min_axis_node_idx < axis_node_range[0]:
                                    axis_node_range[0] = min_axis_node_idx
                                if max_axis_node_idx > axis_node_range[1]:
                                    axis_node_range[1] = max_axis_node_idx
                        if new_range is True:
                            axis_node_ranges.append(axis_node_range)
            for axis_node_range in axis_node_ranges:
                # print('Axis node range:', axis_node_range)
                axis_vol += (axis_node_range[1] - axis_node_range[0]) * side_length * (math.pi * (member_radii ** 2))
            axis_vols.append(axis_vol)
        total_axis_vol = sum(axis_vols)

        # prune member volumes
        # print('Total member vol before pruning:', sum(member_volumes))
        pruned_member_volumes = []
        for idx, member_vol in enumerate(member_volumes):
            if idx not in truss_members_vol_ignore:
                pruned_member_volumes.append(member_vol)
        member_volumes = pruned_member_volumes
        # print('Total member volume:', sum(member_volumes))

        # N. print results
        truss_volume = sum(member_volumes) + sum(node_volumes)
        truss_volume -= sum(intersect_vols)
        truss_volume += total_axis_vol

        # Feasibility constraint is based on the number of intersections
        # invert the constraint such that higher is better (1 is feasible)
        feasibility_constraint = 1 - (num_intersections / self.feasibility_constraint_norm)


        # Augment bit interactions such that it only records interactions where the value is less than the key
        interaction_bit_list = []
        for bit, interactions in bit_interactions.items():
            new_interactions = []
            for interaction in interactions:
                if interaction < bit:
                    new_interactions.append(interaction)
            if len(new_interactions) > 0:
                interaction_bit_list.append(bit)


        # Create autoregressive-safe interaction vector
        interaction_vector = [0 for _ in range(config.num_vars)]
        for bit in interaction_bit_list:
            interaction_vector[bit] = 1

        # print('Truss Volume:', truss_volume / 2.0, truss_volume)
        return (truss_volume / total_volume), feasibility_constraint, interaction_vector



    def get_interaction_vec(self, bit_interactions):
        bit_list = [0 for _ in range(config.num_vars)]
        for bit, interactions in bit_interactions.items():
            bit_list[bit] = 1
        bit_str = ''.join([str(x) for x in bit_list])
        return bit_str

    def calculate_overlap_length(self, edge1, edge2, nodes):
        """
        Calculate the length of the overlapping section between two edges.

        Parameters:
        edge1 (tuple): Pair of node indices defining the first edge.
        edge2 (tuple): Pair of node indices defining the second edge.
        nodes (list of tuple): List of (x, y) coordinates of the nodes.

        Returns:
        float: Length of the overlapping section. Returns 0 if there is no overlap.
        """

        def distance(p1, p2):
            return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

        def is_between(a, c, b):
            # Check if point c is between points a and b
            return a <= c <= b or b <= c <= a

        def point_on_line(p1, p2, p):
            # Check if point p lies on the line segment p1-p2
            return is_between(p1[0], p[0], p2[0]) and is_between(p1[1], p[1], p2[1])

        i1, j1 = edge1
        i2, j2 = edge2
        p1, p2 = nodes[i1], nodes[j1]
        p3, p4 = nodes[i2], nodes[j2]
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4


        # Ensure the lines are parallel by checking the slopes
        # slope1 = (y2 - y1) / (x2 - x1)
        # slope2 = (y4 - y3) / (x4 - x3)
        # Two lines are parallel if their slopes are equal
        if (y2 - y1) * (x4 - x3) != (y4 - y3) * (x2 - x1):
            raise ValueError("The lines are not parallel")

        # Project the line segments onto the x-axis
        line1_proj_x = sorted([x1, x2])
        line2_proj_x = sorted([x3, x4])

        # Calculate the overlap in the x-axis projection
        x_overlap = max(0, min(line1_proj_x[1], line2_proj_x[1]) - max(line1_proj_x[0], line2_proj_x[0]))

        # Project the line segments onto the y-axis
        line1_proj_y = sorted([y1, y2])
        line2_proj_y = sorted([y3, y4])

        # Calculate the overlap in the y-axis projection
        y_overlap = max(0, min(line1_proj_y[1], line2_proj_y[1]) - max(line1_proj_y[0], line2_proj_y[0]))

        # Since the lines are parallel, overlap in one axis is sufficient
        # We return the maximum overlap which should be the same for both projections if they overlap
        return max(x_overlap, y_overlap)



    def remove_parallel_overlaps(self, nodes, edges):
        """
            Remove edges that are parallel and overlapping, retaining the shorter edge.

            Parameters:
            nodes (list of tuple): List of (x, y) coordinates of the nodes.
            edges (list of tuple): List of pairs (i, j) where i and j are indices of nodes being connected.

            Returns:
            list of tuple: Filtered list of edges.
            """

        def calculate_slope_length(edge):
            i, j = edge
            x1, y1 = nodes[i]
            x2, y2 = nodes[j]
            if x2 - x1 == 0:
                slope = float('inf')  # vertical line
            else:
                slope = (y2 - y1) / (x2 - x1)
            length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            return slope, length

        # Create a list to hold the filtered edges
        filtered_edges = edges.copy()
        edges_info = [(edge, *calculate_slope_length(edge)) for edge in edges]

        for i, (edge1, slope1, length1) in enumerate(edges_info):
            parallel_edges = []
            for j, (edge2, slope2, length2) in enumerate(edges_info):
                if i >= j:
                    continue
                if slope1 == slope2:
                    parallel_edges.append(edge2)

                    (i1, j1) = edge1
                    (i2, j2) = edge2
                    # Check if the edges overlap
                    if set([i1, j1]).intersection(set([i2, j2])):
                        # Check the length of the overlap
                        print('Edge 1', edge1)
                        print('Edge 2', edge2)
                        overlap_len = self.calculate_overlap_length(edge1, edge2, nodes)
                        print('Overlap Length:', overlap_len)
                        if overlap_len == 0:
                            continue




                        # Remove the longer edge
                        if length1 > length2:
                            if edge1 in filtered_edges:
                                filtered_edges.remove(edge1)
                        else:
                            if edge2 in filtered_edges:
                                filtered_edges.remove(edge2)

        return filtered_edges




    def get_intersections(self):
        # 4. Account for non-node truss member overlaps
        sidelen = self.sidelen
        # print(self.design_conn_array)
        CA = np.array(self.design_conn_array) - 1
        NC = generateNC(sidelen, self.sidenum)

        print('Before NC:', NC.shape)
        print('Before CA:', CA.shape)

        # visualize_graph(NC.tolist(), CA.tolist())
        NC, CA, found = add_intersection_nodes(NC.tolist(), CA.tolist())
        # visualize_graph(NC, CA)
        # exit(0)

        for x in range(100):
            print('Iteration:', x)
            NC, CA, found = add_intersection_nodes(NC, CA)
            if found is False:
                break
        # visualize_graph(NC, CA)



        CA = np.array(CA) + 1
        return NC, CA






if __name__ == '__main__':
    sidenum = 3
    num_vars = TrussFeatures.get_member_count_repeatable(sidenum)

    # bit_str = '101100001010000000010100101100'
    bit_str =   '100000000000000000000000000000'
    bit_list = [int(x) for x in bit_str]





    engine = TrussVolumeFraction(sidenum, bit_list)
    results = engine.evaluate(member_radii=0.1, side_length=1.0 / 2)
    # engine.visualize()

    vol_frac, feasibility_constraint, int_vector = results
    print('Volume Fraction:', vol_frac)



    int_vector_str = ''.join([str(x) for x in int_vector])


    print('-----------> Bit list:', bit_str)
    print('---> Bit Interactions:', int_vector_str)
    # print('--> Full Interactions:', engine.get_interaction_vec(bit_interactions))







