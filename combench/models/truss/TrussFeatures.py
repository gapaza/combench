import math
import matplotlib.pyplot as plt
import config
import textwrap



# node_positions = {
#     1: (0, 0),  # Bottom left
#     2: (0, 1),  # Middle left
#     3: (0, 2),  # Top left
#     4: (1, 0),  # Bottom middle
#     5: (1, 1),  # Middle
#     6: (1, 2),  # Top middle
#     7: (2, 0),  # Bottom right
#     8: (2, 1),  # Middle right
#     9: (2, 2)   # Top right
# }
node_positions = {}
idx = 1
for x in range(config.sidenum):
    for y in range(config.sidenum):
        node_positions[idx] = (x, y)
        idx += 1



class TrussFeatures:


    def __init__(self, bit_list, sidenum, problem):
        self.bit_list = bit_list
        self.bool_list = [True if x == 1 else False for x in bit_list]
        self.sidenum = sidenum
        self.problem = problem

        # Get the top edge nodes
        self.top_nodes = TrussFeatures.get_top_edge_nodes(self.sidenum)

        # Create the repeatable connectivity array
        self.connectivity_array_repeatable = TrussFeatures.get_connectivity_array_repeatable(self.sidenum)

        # Create the connectivity array
        self.connectivity_array = TrussFeatures.get_connectivity_array(self.sidenum)

        # Create the complete boolean array list
        self.complete_boolean_array_list = TrussFeatures.calc_design_boolean_array(self.sidenum, self.top_nodes, self.bool_list, self.connectivity_array_repeatable)

        # Create the design connectivity array (list of lists of 2 integers)
        self.design_conn_array = TrussFeatures.calc_design_connectivity_array(self.connectivity_array, self.complete_boolean_array_list)


    def visualize_design(self, prompt=None, temp=None):
        # bit_str = ''.join([str(bit) for bit in self.bit_list])

        # Plot the nodes
        for node, pos in node_positions.items():
            plt.plot(pos[0], pos[1], 'o', markersize=10, color='b')  # 'o' for circle markers
            plt.annotate(str(node), (pos[0], pos[1]), textcoords="offset points", xytext=(5, 5), ha='center')

        # Draw truss members
        for connection in self.design_conn_array:
            node1, node2 = connection
            pos1, pos2 = node_positions[node1], node_positions[node2]
            plt.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'k-')  # 'k-' for black lines

        # Set plot limits and labels
        plt.xlim(-1, config.sidenum)
        plt.ylim(-1, config.sidenum)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        if temp is not None:
            title = '3x3 Truss Structure Visualization ('+str(temp)+')\n'  # + bit_str
        else:
            title = '3x3 Truss Structure Visualization\n'  # + bit_str
        plt.title(title)
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')

        # Add a text box outside the plot
        if prompt is not None:
            wrapped_text = textwrap.fill(prompt, width=50)
            plt.figtext(
                0.5, 0.05,
                wrapped_text,
                ha='center', va='bottom', fontsize=10,
                bbox=dict(facecolor='red', alpha=0.5)
            )
            plt.subplots_adjust(bottom=0.3)


        plt.savefig('truss.png')

        plt.show()

    def print_components(self):
        print(self.sidenum)
        print(self.top_nodes)
        print(len(self.connectivity_array_repeatable), self.connectivity_array_repeatable)
        print(len(self.connectivity_array), self.connectivity_array)
        print(len(self.complete_boolean_array_list), self.complete_boolean_array_list)
        print(len(self.design_conn_array), self.design_conn_array)

    @staticmethod
    def calc_design_connectivity_array(connectivity_array, design_boolean_array):
        # Filter the connectivity array based on the design boolean array
        filtered_conn_array = [connectivity_array[index] for index, decision in enumerate(design_boolean_array) if decision]
        return filtered_conn_array

    # ------------------------------------
    # Design Boolean Arrays
    # ------------------------------------

    @staticmethod
    def calc_design_boolean_array(sidenum, top_nodes, design_boolean_array_repeatable, connectivity_array_repeatable):
        complete_boolean_array_list = []
        member_index = 0
        sidenum_squared = int(sidenum * sidenum)

        for i in range(sidenum_squared - 1):
            node = i + 1
            for j in range(node + 1, sidenum_squared + 1):
                right_edge = False
                top_edge = False
                if node > (sidenum_squared - sidenum):  # Identifying right edge members
                    if j > (sidenum_squared - sidenum):
                        right_edge = True
                        # Corresponding left edge member
                        repeated_member = [int(node - ((sidenum - 1) * sidenum)), int(j - ((sidenum - 1) * sidenum))]
                        repeated_member_index = TrussFeatures.get_member_index(connectivity_array_repeatable, repeated_member)
                        complete_boolean_array_list.append(design_boolean_array_repeatable[repeated_member_index])
                if node in top_nodes:  # Identifying top edge members
                    if j in top_nodes:
                        top_edge = True
                        # Corresponding bottom edge member
                        repeated_member = [int(node - (sidenum - 1)), int(j - (sidenum - 1))]
                        repeated_member_index = TrussFeatures.get_member_index(connectivity_array_repeatable, repeated_member)
                        complete_boolean_array_list.append(design_boolean_array_repeatable[repeated_member_index])
                if not right_edge and not top_edge:
                    complete_boolean_array_list.append(design_boolean_array_repeatable[member_index])
                    member_index += 1

        return complete_boolean_array_list

    # ------------------------------------
    # Connectivity Arrays
    # ------------------------------------

    @staticmethod
    def get_connectivity_array(sidenum):
        # Get the number of possible truss members
        total_number_of_members = TrussFeatures.get_member_count(sidenum)

        # Create the connectivity array
        member_count = 0
        complete_connectivity_array = [[0, 0] for _ in range(total_number_of_members)]
        sidenum_squared = int(sidenum * sidenum)
        for i in range(sidenum_squared - 1):
            for j in range(i + 1, sidenum_squared):
                complete_connectivity_array[member_count][0] = i + 1
                complete_connectivity_array[member_count][1] = j + 1
                member_count += 1

        return complete_connectivity_array

    @staticmethod
    def get_connectivity_array_repeatable(sidenum):
        member_count = 0
        number_of_variables = TrussFeatures.get_member_count_repeatable(sidenum)

        # Get the top edge nodes
        top_nodes = TrussFeatures.get_top_edge_nodes(sidenum)  # [3, 6, 9]

        # Create the repeatable connectivity array
        repeatable_connectivity_array = [[0, 0] for _ in range(number_of_variables)]
        sidenum_squared = int(sidenum * sidenum)
        for i in range(sidenum_squared - 1):  # iterates over 9 nodes
            node = i + 1  # 1 - 9
            for j in range(node + 1, sidenum_squared + 1):  # 2 - 9, 3 - 9, 4 - 9, ..., 8 - 9
                if node > (sidenum_squared - sidenum):  # identifying right edge members (if node > 6)
                    if j > (sidenum_squared - sidenum):  # if j > 6
                        continue
                if (node in top_nodes):  # identifying top edge members (if node in [3, 6, 9])
                    if (j in top_nodes):  # if j in [3, 6, 9]
                        continue
                repeatable_connectivity_array[member_count][0] = i + 1
                repeatable_connectivity_array[member_count][1] = j
                member_count += 1

        return repeatable_connectivity_array

    # ------------------------------------
    # Number of Members
    # ------------------------------------

    @staticmethod
    def get_member_count(sidenum):
        if sidenum >= 5:
            sidenum_squared = int(sidenum * sidenum)
            total_number_of_members = sidenum_squared * (sidenum_squared - 1) // 2
        else:
            # this is 36 for 3x3
            total_number_of_members = (math.factorial(int(sidenum * sidenum)) //
                                       (math.factorial(int(sidenum * sidenum) - 2) * math.factorial(2)))
        return total_number_of_members

    @staticmethod
    def get_member_count_repeatable(sidenum):
        # Get total variables
        total_number_of_members = TrussFeatures.get_member_count(sidenum)

        # Get repeatable variables
        number_of_repeatable_members = (2 * math.factorial(int(sidenum)) //
                                        (math.factorial(int(sidenum - 2)) * math.factorial(2)))

        # Subtract and return
        return total_number_of_members - number_of_repeatable_members

    # ------------------------------------
    # Top Nodes
    # ------------------------------------

    @staticmethod
    def get_top_edge_nodes(sidenum):
        top_nodes = []
        reached_right_edge = False
        node = int(sidenum)  # 3
        while not reached_right_edge:
            if node > (sidenum * sidenum):
                reached_right_edge = True
            else:
                top_nodes.append(node)
                node += int(sidenum)
        return top_nodes

    # ------------------------------------
    # Get member index
    # ------------------------------------

    @staticmethod
    def get_member_index(connectivity_array, member):
        for i in range(len(connectivity_array)):
            if connectivity_array[i][0] == member[0] and connectivity_array[i][1] == member[1]:
                return i
        return -1  # Return -1 if the member is not found

    # ------------------------------------
    # Calculate volume fraction
    # ------------------------------------

    def calculate_volume_fraction_5x5(self, member_radii=50e-6, side_length=100e-5):
        # member_radii = 50e-6
        # side_length = 100e-3

        # 1. calculate total volume of truss
        depth = member_radii * 2
        width = side_length * 5
        height = side_length * 5
        total_volume = depth * width * height

        # 2. calculate total volume of truss members
        member_volumes = []
        member_lengths = []
        member_positions = []
        for connection in self.design_conn_array:
            node1, node2 = connection
            pos1, pos2 = node_positions[node1], node_positions[node2]
            pos1 = [x * side_length for x in pos1]
            pos2 = [x * side_length for x in pos2]
            member_positions.append((pos1, pos2))
            length = math.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
            member_lengths.append(length)
            member_volume = math.pi * member_radii**2 * length
            member_volumes.append(member_volume)

        # 3. Account for truss member overlaps at nodes
        node_volumes = []
        node_volume = (4/3) * math.pi * member_radii**3
        sub_vol = math.pi * member_radii**2 * member_radii
        for x in range(1, 26):
            connected_members_indices = [i for i, connection in enumerate(self.design_conn_array) if x in connection]
            if len(connected_members_indices) > 0:
                node_volumes.append(node_volume)
            for idx in connected_members_indices:
                member_volumes[idx] -= sub_vol

        # 4. Account for non-node truss member overlaps
        intersect_vols = []
        for member_a_idx in range(len(self.design_conn_array)):
            member_a_n1, member_a_n2 = self.design_conn_array[member_a_idx]
            a_pos_1, a_pos_2 = node_positions[member_a_n1], node_positions[member_a_n2]
            for member_b_idx in range(member_a_idx + 1, len(self.design_conn_array)):
                member_b_n1, member_b_n2 = self.design_conn_array[member_b_idx]
                b_pos_1, b_pos_2 = node_positions[member_b_n1], node_positions[member_b_n2]
                intersects = intersect(a_pos_1, a_pos_2, b_pos_1, b_pos_2)
                if intersects is True:
                    angle = calculate_angle(a_pos_1, a_pos_2, b_pos_1, b_pos_2)
                    volume = estimate_intersection_volume(member_radii, angle)
                    intersect_vols.append(volume)
                    # print(intersects, angle, volume)
        # print('Intersecting volumes:', sum(intersect_vols))


        # 5. Account for parallel truss member overlaps
        x_axis_1_nodes = [1, 2, 3, 4, 5]
        x_axis_2_nodes = [6, 7, 8, 9, 10]
        x_axis_3_nodes = [11, 12, 13, 14, 15]
        x_axis_4_nodes = [16, 17, 18, 19, 20]
        x_axis_5_nodes = [21, 22, 23, 24, 25]
        y_axis_1_nodes = [1, 6, 11, 16, 21]
        y_axis_2_nodes = [2, 7, 12, 17, 22]
        y_axis_3_nodes = [3, 8, 13, 18, 23]
        y_axis_4_nodes = [4, 9, 14, 19, 24]
        y_axis_5_nodes = [5, 10, 15, 20, 25]
        x_axis_nodes = [x_axis_1_nodes, x_axis_2_nodes, x_axis_3_nodes, x_axis_4_nodes, x_axis_5_nodes]
        y_axis_nodes = [y_axis_1_nodes, y_axis_2_nodes, y_axis_3_nodes, y_axis_4_nodes, y_axis_5_nodes]
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
        # print('Total axis vol:', total_axis_vol)


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

        # print(member_volumes)
        # print(self.design_conn_array)
        # print('Total truss volume:', truss_volume)
        # print('Total volume:', total_volume)
        # print('Volume fraction:', truss_volume / total_volume)

        return truss_volume / total_volume





def ccw(A, B, C):
    """Check if points A, B, and C are counter-clockwise."""
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def intersect(A, B, C, D):
    # cannot share a common node
    if A == C or A == D or B == C or B == D:
        return False
    """Check if line segments AB and CD intersect."""
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def calculate_angle(A, B, C, D):
    """
    Calculate the angle between vectors AB and CD.
    """

    def vector(a, b):
        return (b[0] - a[0], b[1] - a[1])

    def dot_product(u, v):
        return u[0] * v[0] + u[1] * v[1]

    def magnitude(v):
        return math.sqrt(v[0] ** 2 + v[1] ** 2)

    u = vector(A, B)
    v = vector(C, D)

    angle_rad = math.acos(dot_product(u, v) / (magnitude(u) * magnitude(v)))
    return math.degrees(angle_rad)

def estimate_intersection_volume(radius, angle_degrees):
    """
    Estimates the volume of intersection between two cylindrical truss members
    based on their radius and the angle of intersection.

    Parameters:
    - radius: The radius of the truss members (assuming both have equal radii).
    - angle_degrees: The angle of intersection between the truss members in degrees.

    Returns:
    - The estimated volume of intersection.
    """
    # Convert angle from degrees to radians for calculations
    angle_radians = math.radians(angle_degrees)

    # Simplistic approach to estimate "height" based on angle; more accurate methods may vary
    # This assumes a larger angle results in a smaller intersection volume and vice versa
    h = radius * math.sin(angle_radians / 2)

    # Estimate intersection volume as a spherical cap volume
    V = (math.pi * h**2) / 3 * (3*radius - h)

    return V

def get_connections_list(sidenum):
    problem = None
    bits = TrussFeatures.get_member_count_repeatable(sidenum)
    bit_connections_list = []
    for i in range(bits):
        bit_list = [0 for _ in range(bits)]
        bit_list[i] = 1
        arch = TrussFeatures(bit_list, sidenum, problem)
        bit_connections_list.append(arch.design_conn_array)
    return bit_connections_list







if __name__ == '__main__':

    member_count = TrussFeatures.get_member_count_repeatable(7)
    member_count_dict = {}
    for x in range(3, 21):
        member_count = TrussFeatures.get_member_count_repeatable(x)
        member_count_dict[x] = member_count
    print(member_count_dict)
    exit(0)



    design = '101000001110000010010100111001'
    design = [int(x) for x in design]
    sidenum = 3
    problem = None
    arch = TrussFeatures(design, sidenum, problem)
    arch.visualize_design()
    print(arch.design_conn_array)
    exit(0)





    # connections_list = get_connections_list()
    # print(connections_list)

    # 0.006013693939011665


    # 0.25010660708917704 (no side length scaling)
    # 0.025010660708917762 (side length scaling)



    sidenum = 5
    problem = None


    connections_3x3 = get_connections_list(3)


    connections_4x4 = get_connections_list(4)
    connections_5x5 = get_connections_list(5)
    connections_6x6 = get_connections_list(6)

    print(connections_3x3)
    exit(0)




    # member_count = TrussFeatures.get_member_count_repeatable(3)
    # print(member_count)  # 30
    # member_count = TrussFeatures.get_member_count_repeatable(4)
    # print(member_count)  # 108
    # member_count = TrussFeatures.get_member_count_repeatable(5)
    # print(member_count)  # 280
    # member_count = TrussFeatures.get_member_count_repeatable(6)
    # print(member_count)  # 600





    # design = [0 for x in range(config.num_vars)]
    # design[0] = 1
    # design[1] = 1
    # design[6] = 1
    # design[8] = 1
    # design[42] = 1
    # design[260] = 1

    # design = '0000001010100000000100000000011011110001000000010001000100000000001000000001000000000001111001000000010010000011001010000010000000111100000111010000000000000111100100000100000000000000000010001000000000010000001000000000000000000000000001000000000010000010100110000000000010010000'
    # design = [int(x) for x in design]

    # design = [1 for x in range(config.num_vars)]
    # for x in range(10, 15):
    #     design[x] = 1


    # design_str = ''.join([str(x) for x in design])
    #
    #
    # arch = TrussFeatures(design, sidenum, problem)
    # arch.visualize_design()
    #
    # arch.calculate_volume_fraction_5x5()





