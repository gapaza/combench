sidenum_nvar_map = {2: 6, 3: 30, 4: 108, 5: 280, 6: 600, 7: 1134, 8: 1960, 9: 3168, 10: 4860, 11: 7150, 12: 10164, 13: 14040, 14: 18928, 15: 24990, 16: 32400, 17: 41344, 18: 52020, 19: 64638, 20: 79420}

problem1 = {
    'name': 'truss-3x3-problem1',
    'sidenum': 3,
    'y_modulus': 18162.0,
    'member_radii': 0.1,
    'side_length': 3,
}



from combench.models.truss import representation as rep


# ----------------- PROBLEMS ----------------- #
from combench.models.truss.problems.cantilever import Cantilever

# problem = Cantilever.get_problem(
#     None,
#     x_range=3,
#     y_range=3,
#     x_res=3,
#     y_res=3,
#     radii=0.2,
#     y_modulus=210e9
# )

param_dict = {
    'x_range': 3,
    'y_range': 3,
    'x_res': 3,
    'y_res': 3,
    'radii': 0.2,
    'y_modulus': 210e9
}

problem = Cantilever.type_1(
    param_dict
)



# ----------------- EVALUATION FUNCTIONS ----------------- #
from combench.models.truss.vol.c_geometry import vox_space
# from truss.model.stiffness.truss_model_old import eval_load_cond
from combench.models.truss.stiffness2.truss_model import eval_load_cond

def eval_stiffness(problem, design_rep, normalize=False):

    # 1. Validate load conditions
    if 'load_conds' not in problem:
        raise ValueError('Problem must have nodes_loads')

    # 2. Validate at least one member
    bit_list, bit_str, node_idx_pairs, node_coords = rep.convert(problem, design_rep)
    if 1 not in bit_list:
        return [0 for x in range(len(problem['load_conds']))]

    # 3. Validate no overlapping members
    design_rep_no = rep.remove_overlapping_members(problem, design_rep)
    if sum(design_rep_no) < sum(bit_list):
        # print('ERROR: Overlapping members in design representation')
        # return [0 for x in range(len(problem['load_conds']))]
        design_rep = design_rep_no

    # 4. Evaluate stiffness for each load condition
    stiff_vals = []
    for load_conds in problem['load_conds']:
        stiff = eval_load_cond(problem, design_rep, load_conds)
        stiff_vals.append(stiff)

    # 5. Normalize stiffness values
    if normalize is True:
        norms = set_norms(problem)
        norm_stiff_vals = []
        for norm, stiff in zip(norms, stiff_vals):
            if norm != 0:
                norm_val = stiff/norm
                if norm_val > 1.0:  # If stiffness greater than fully connected, due to numerical issues
                    norm_stiff_vals.append(0.0)
                elif norm_val < 0.0:
                    norm_stiff_vals.append(0.0)
                else:
                    norm_stiff_vals.append(stiff/norm)
            else:
                raise ValueError('Norms cannot be zero', norms)
        stiff_vals = norm_stiff_vals
    return stiff_vals

def set_norms(problem):
    if 'norms' in problem:
        return problem['norms']
    else:
        print('--> Calculating norms for problem')
        fully_connected = [1 for x in range(rep.get_num_bits(problem))]
        fully_connected_no = rep.remove_overlapping_members(problem, fully_connected)
        stiff_vals_raw = eval_stiffness(problem, fully_connected_no, normalize=False)
        stiff_vals = []
        for sv in stiff_vals_raw:
            if sv != 0:
                stiff_vals.append(sv * 1.2)  # Add margin
            else:
                stiff_vals.append(1)
        problem['norms'] = stiff_vals
        return stiff_vals

def eval_volfrac(problem, design_rep):
    bit_list, bit_str, node_idx_pairs, node_coords = rep.convert(problem, design_rep)
    vol_frac = vox_space(problem, node_idx_pairs, resolution=100)
    return vol_frac


if __name__ == '__main__':

    num_bits = rep.get_num_bits(problem)
    design_rep =[1 for x in range(num_bits)]
    print(rep.get_design_metrics(problem, design_rep))
    rep.viz(problem, design_rep, 'curr_problem.png')

























