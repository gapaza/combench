from combench.models.truss.stiffness.generateC import generateC
from combench.models.truss.stiffness.generateNC import generateNC
from combench.models.truss.stiffness.modifyAreas import modifyAreas
import numpy as np

from combench.models.truss.TrussFeatures import TrussFeatures





class TrussStiffness:


    def __init__(self):
        sidenum = 3  # For a 3x3 grid
        sel = 0.01  # Unit cell size

        # CA = np.array([
        #     [1, 2], [2, 3], [1, 4], [1, 5], [2, 5], [3, 5], [3, 6], [4, 5], [5, 6],
        #     [4, 7], [5, 7], [5, 8], [5, 9], [6, 9], [7, 8], [8, 9]
        # ])  # Connectivity array
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
        member_radii = 250e-6  # Radii of truss elements
        y_modulus = 1.8162e6

        c11, c22, stiffness_ratio = TrussStiffness.evaluate(CA, sidenum, sel, member_radii, y_modulus)


    @staticmethod
    def evaluate(CA_list, sidenum, sidelen, member_radii, y_modulus):
        CA = np.array(CA_list)
        rvar = member_radii * np.ones(CA.shape[0])  # Radii of truss elements

        C = TrussStiffness.get_stiffness_tensor(sidenum, sidelen, rvar, y_modulus, CA)
        if C is None:
            return None
        c11_vstiff = C[0, 0]  # Vertical stiffness
        c22_hstiff = C[1, 1]  # Horizontal stiffness
        stiffness_ratio = 0
        if c22_hstiff != 0:
            stiffness_ratio = c11_vstiff / c22_hstiff
        return c11_vstiff, c22_hstiff, stiffness_ratio

    @staticmethod
    def evaluate_decomp(CA_list, sidenum, sidelen, member_radii, y_modulus, new_nodes):
        CA = np.array(CA_list)
        rvar = member_radii * np.ones(CA.shape[0])  # Radii of truss elements
        NC = np.array(new_nodes)
        C = TrussStiffness.get_stiffness_tensor(sidenum, sidelen, rvar, y_modulus, CA, NC=NC)
        if C is None:
            return None
        c11_vstiff = C[0, 0]  # Vertical stiffness
        c22_hstiff = C[1, 1]  # Horizontal stiffness
        stiffness_ratio = 0
        if c22_hstiff != 0:
            stiffness_ratio = c11_vstiff / c22_hstiff
        return c11_vstiff, c22_hstiff, stiffness_ratio



    # NOTE: sel MUST be the total length of the truss side
    @staticmethod
    def get_stiffness_tensor(sidenum, sel, rvar, E, CA, NC=None):

        # Generate vector with nodal coordinates
        if NC is None:
            NC = generateNC(sel, sidenum)

        # Calculate Avar & modify for edge members
        # print('CA: ', CA)
        # print('NC: ', NC)

        Avar = np.pi * (rvar ** 2)  # Cross-sectional areas of truss members
        Avar = modifyAreas(Avar, CA, NC, sidenum)

        # Initialize C matrix
        C = np.zeros((3, 3))  # Assuming 3x3 for 2D truss analysis

        # Develop C-matrix from K-matrix

        # Print all generateC input values
        # print("sel: ", sel)
        # print("rvar: ", rvar)
        # print("NC: ", NC)
        # print("CA: ", CA)
        # print("Avar: ", Avar)
        # print("E: ", E)
        # print("C: ", C)
        # print("sidenum: ", sidenum)

        results = generateC(sel, rvar, NC, CA, Avar, E, C, sidenum)
        if results is None:
            return None
        else:
            C, uBasket, fBasket = results
            return C


    @staticmethod
    def decompose_for_intersections(CA, NC):
        # Find overlaps

        pass





if __name__ == "__main__":
    # sidenum = 3
    # design = '111111111111111111111111111111'
    # # design = [1 for x in range(0, 600)]
    # unit_len = 0.01
    # y_modulus = 1.8162e6
    # member_rads = 250e-6
    #
    # results = TrussStiffness.evaluate(design, sidenum, unit_len, y_modulus, member_rads)


    TrussStiffness()
