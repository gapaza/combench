import config
import time
import math


from combench.models.truss.vol.TrussVolumeFraction import TrussVolumeFraction
from combench.models.truss.stiffness.TrussStiffness import TrussStiffness
from combench.models.truss import sidenum_nvar_map
from combench.interfaces.model import Model
from combench.models.utils import random_binary_design


class TrussModel(Model):

    def __init__(self, problem_formulation):
        super().__init__(problem_formulation)
        self.sidenum = problem_formulation['sidenum']
        self.y_modulus = problem_formulation['y_modulus']
        self.member_radii = problem_formulation['member_radii']
        self.side_length = problem_formulation['side_length']
        self.norms = self.load_norms()
        print('Norm values: {}'.format(self.norms))

    def load_norms(self):
        if 'norms' in self.problem_store:
            return self.problem_store['norms']

        # Calculate the norms
        num_bits = sidenum_nvar_map[self.sidenum]
        random_design = [1 for x in range(num_bits)]
        evals = []
        evals.append(self.evaluate(random_design, normalize=False))
        max_vstiff = min([evals[i][0] for i in range(len(evals))])
        max_volume_fraction = max([evals[i][1] for i in range(len(evals))])
        max_vstiff_norm = abs(max_vstiff) * 1.1  # Add margin
        max_volume_fraction_norm = max_volume_fraction * 1.1  # Add margin
        self.problem_store['norms'] = [max_vstiff_norm, max_volume_fraction_norm]
        self.save_problem_store()
        return [max_vstiff_norm, max_volume_fraction_norm]

    def random_design(self):
        num_bits = sidenum_nvar_map[self.sidenum]
        return random_binary_design(num_bits)

    def evaluate(self, design, normalize=True):
        y_modulus = self.y_modulus
        member_radii = self.member_radii
        side_length = self.side_length
        v_stiff, volume_fraction, stiff_ratio = self._evaluate(design, y_modulus, member_radii, side_length)

        if normalize is True:
            if v_stiff == 0 and volume_fraction == 1:
                return 0, 1
            else:
                v_stiff_norm, volume_fraction_norm = self.load_norms()
                v_stiff /= v_stiff_norm
                volume_fraction /= volume_fraction_norm
        return v_stiff, volume_fraction

    def _evaluate(self, design_array, y_modulus, member_radii, side_length):

        # 1. Calculate the volume fraction
        curr_time = time.time()
        vf_client = TrussVolumeFraction(self.sidenum, design_array, side_length=side_length)
        design_conn_array = vf_client.design_conn_array
        volume_fraction, feasibility_constraint, interaction_list = vf_client.evaluate(member_radii, side_length)
        # volume_fraction, feasibility_constraint, interaction_list = 0, 0, 0

        # print("Time taken for volume fraction: ", time.time() - curr_time)

        # 2. Calculate the stiffness
        curr_time = time.time()
        results = TrussStiffness.evaluate(design_conn_array, self.sidenum, side_length, member_radii, y_modulus)
        if results is None:
            return 0, 1, 0
        v_stiff, h_stiff, stiff_ratio = results

        # Check if v_stiff is numpy nan
        if math.isnan(v_stiff):
            return 0, 1, 0

        return -v_stiff, volume_fraction, stiff_ratio

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
        results = TrussStiffness.evaluate_decomp(new_design_conn_array, self.sidenum, side_length, member_radii, y_modulus, new_nodes)
        if results is None:
            return 0, 0, 0, 0, 0
        v_stiff, h_stiff, stiff_ratio = results
        print("Time taken for stiffness: ", time.time() - curr_time)

        return  v_stiff, h_stiff, stiff_ratio




from combench.models.truss import problem1

if __name__ == '__main__':


    truss_model = TrussModel(problem1)


    # design_str = '111010000010100010100000101010'
    # design_array = [int(bit) for bit in design_str]
    design_array = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

    curr_time = time.time()
    objectives = truss_model.evaluate(design_array)
    print('Objectives:', objectives)
    print("Time taken: ", time.time() - curr_time)
