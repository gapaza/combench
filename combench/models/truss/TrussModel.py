import config
import time


from combench.models.truss.vol.TrussVolumeFraction import TrussVolumeFraction
from combench.models.truss.stiffness.TrussStiffness import TrussStiffness

from combench.interfaces.model import Model


class TrussModel(Model):

    def __init__(self, problem_formulation):
        super().__init__(problem_formulation)
        self.sidenum = problem_formulation['sidenum']

    def evaluate(self, design_array, y_modulus, member_radii, side_length):

        # 1. Calculate the volume fraction
        curr_time = time.time()
        vf_client = TrussVolumeFraction(self.sidenum, design_array, side_length=side_length)
        design_conn_array = vf_client.design_conn_array
        volume_fraction, feasibility_constraint, interaction_list = vf_client.evaluate(member_radii, side_length)
        # volume_fraction, feasibility_constraint, interaction_list = 0, 0, 0

        # print("Time taken for volume fraction: ", time.time() - curr_time)

        # 2. Calculate the stiffness
        curr_time = time.time()
        v_stiff, h_stiff, stiff_ratio = TrussStiffness.evaluate(design_conn_array, self.sidenum, side_length, member_radii, y_modulus)
        # print("Time taken for stiffness: ", time.time() - curr_time)

        # print("Volume fraction: ", volume_fraction)
        # print("Vertical stiffness: ", v_stiff)
        # print("Horizontal stiffness: ", h_stiff)
        # print("Stiffness ratio: ", stiff_ratio)
        # print("Feasibility constraint: ", feasibility_constraint)

        return v_stiff, h_stiff, stiff_ratio, volume_fraction, feasibility_constraint

    def evaluate_decomp(self, design_array, y_modulus, member_radii, side_length):

        # 1. Calculate the volume fraction
        curr_time = time.time()
        vf_client = TrussVolumeFraction(self.sidenum, design_array, side_length=side_length)
        design_conn_array = vf_client.design_conn_array
        volume_fraction, feasibility_constraint, interaction_list = vf_client.evaluate(member_radii, side_length)
        new_nodes, new_design_conn_array = vf_client.get_intersections()
        # volume_fraction, feasibility_constraint, interaction_list = 0, 0, 0

        # print("Time taken for volume fraction: ", time.time() - curr_time)

        # 2. Calculate the stiffness
        curr_time = time.time()
        v_stiff, h_stiff, stiff_ratio = TrussStiffness.evaluate_decomp(new_design_conn_array, self.sidenum, side_length, member_radii, y_modulus, new_nodes)
        print("Time taken for stiffness: ", time.time() - curr_time)

        # print("Volume fraction: ", volume_fraction)
        # print("Vertical stiffness: ", v_stiff)
        # print("Horizontal stiffness: ", h_stiff)
        # print("Stiffness ratio: ", stiff_ratio)
        # print("Feasibility constraint: ", feasibility_constraint)

        return v_stiff, h_stiff, stiff_ratio, volume_fraction, feasibility_constraint



if __name__ == '__main__':

    problem_formulation = {
        'sidenum': config.sidenum
    }
    truss_model = TrussModel(problem_formulation)




    design_str = '111010000010100010100000101010'
    design_array = [int(bit) for bit in design_str]


    y_modulus = 18162.0
    member_radii = 0.1
    side_length = 3

    curr_time = time.time()
    objectives = truss_model.evaluate(design_array, y_modulus, member_radii, side_length)
    # print('\n----------')
    # truss_model.evaluate_decomp(design_array, y_modulus, member_radii, side_length)
    print('Objectives:', objectives)
    print("Time taken: ", time.time() - curr_time)
