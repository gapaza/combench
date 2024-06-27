from combench.models.truss import representation as rep
import random

################################################
# ----------------- PROBLEMS ----------------- #
################################################

from combench.models.truss.problems.cantilever import Cantilever
from combench.models.truss.problems.truss_type_1 import TrussType1

# 1. Define Problems
N = 4
param_dict = {
    'x_range': N,
    'y_range': N,
    'x_res': N,
    'y_res': N,
    'radii': 0.2,
    'y_modulus': 210e9
}

# 2. Generate Problem Set, Sample
problem_set = TrussType1.enumerate(
    param_dict
)
random.seed(4)
problem_set = random.sample(problem_set, 64)
# for p in problem_set:
#     set_norms(p)

# 3. Split into Training and Testing
val_problem_indices = [0, 1, 2, 3]
train_problems = [problem_set[i] for i in range(len(problem_set)) if i not in val_problem_indices]
val_problems = [problem_set[i] for i in val_problem_indices]

# 4. Get individual problems for testing
train_problem = train_problems[0]
val_problem = val_problems[0]




############################################################
# ----------------- EVALUATION FUNCTIONS ----------------- #
############################################################

REPAIR_OVERLAPS = True
FC_MARGIN = 1.0

VOLFRAC_RESOLUTION = 50
VOLFRAC_MIN, VOLFRAC_MAX = 0.2, 1.0

from combench.models.truss.vol.c_geometry import vox_space
from combench.models.truss.stiffness.truss_model import eval_load_cond

def eval_stiffness(problem, design_rep, normalize=False, verbose=False):

    # 1. Validate load conditions
    if 'load_conds' not in problem:
        raise ValueError('Problem must have nodes_loads')

    # 2.1 Validate at least one member
    bit_list, bit_str, node_idx_pairs, node_coords = rep.convert(problem, design_rep)
    if 1 not in bit_list:
        if verbose is True:
            return [0 for x in range(len(problem['load_conds']))], []
        else:
            return [0 for x in range(len(problem['load_conds']))]

    # 2.2 Validate no single node connections
    used_nodes_idx = rep.get_used_nodes(problem, bit_list)
    fixed_nodes = rep.get_all_fixed_nodes(problem)
    free_used_nodes = [node for node in used_nodes_idx if node not in fixed_nodes]
    for node_idx in free_used_nodes:
        node_connections = rep.get_node_connections(problem, design_rep, node_idx)
        if len(node_connections) <= 1:
            if verbose is True:
                return [0 for x in range(len(problem['load_conds']))], []
            else:
                return [0 for x in range(len(problem['load_conds']))]

    # 3. Validate no overlapping members
    if REPAIR_OVERLAPS is True:
        design_rep_no = rep.remove_overlapping_members(problem, design_rep)
        if sum(design_rep_no) < sum(bit_list):
            # print('ERROR: Overlapping members in design representation')
            # return [0 for x in range(len(problem['load_conds']))]
            design_rep = design_rep_no

    # 4. Evaluate stiffness for each load condition
    stiff_vals = []
    stiff_model_info = []
    for load_conds in problem['load_conds']:
        stiff, extra_info = eval_load_cond(problem, design_rep, load_conds)
        stiff_vals.append(stiff)
        stiff_model_info.append(extra_info)

    # 5. Normalize stiffness values
    if normalize is True:
        set_norms(problem)
        norms = problem['norms']
        norm_stiff_vals = []
        for norm, stiff in zip(norms, stiff_vals):
            if norm != 0:
                norm_val = stiff/norm
                # if norm_val > 10.0:  # If stiffness greater than fully connected, due to numerical issues
                #     norm_stiff_vals.append(0.0)
                if norm_val < 0.0:
                    norm_stiff_vals.append(0.0)
                else:
                    norm_stiff_vals.append(stiff/norm)
            else:
                raise ValueError('Norms cannot be zero', norms)
        stiff_vals = norm_stiff_vals

    if verbose is True:
        return stiff_vals, stiff_model_info
    else:
        return stiff_vals

def set_norms(problem):
    if 'norms' not in problem:

        # 1. Stiffness norms
        fully_connected = [1 for x in range(rep.get_num_bits(problem))]
        if REPAIR_OVERLAPS is True:
            fully_connected_no = rep.remove_overlapping_members(problem, fully_connected)
            fully_connected = fully_connected_no
        stiff_vals_raw = eval_stiffness(problem, fully_connected, normalize=False)
        stiff_vals = []
        for sv in stiff_vals_raw:
            if sv != 0:
                stiff_vals.append(sv * FC_MARGIN)  # Add margin
            else:
                stiff_vals.append(1)
        problem['norms'] = stiff_vals

        # 2. Volfrac norms
        volfrac_max = VOLFRAC_MAX
        if volfrac_max is None:
            volfrac_max = eval_volfrac(problem, fully_connected, normalize=False)

        volfrac_min = VOLFRAC_MIN
        if volfrac_min is None:
            volfrac_min = 0.0  # Hardcode for now
        problem['volfrac_norms'] = [volfrac_min, volfrac_max]
        print('--> Problem Normalization:', stiff_vals, [volfrac_min, volfrac_max])




def eval_volfrac(problem, design_rep, normalize=False):
    bit_list, bit_str, node_idx_pairs, node_coords = rep.convert(problem, design_rep)
    if 1 not in bit_list:
        return 0

    if REPAIR_OVERLAPS is True:
        design_rep_no = rep.remove_overlapping_members(problem, design_rep)
        bit_list_no, bit_str_no, node_idx_pairs_no, node_coords_no = rep.convert(problem, design_rep_no)
        node_idx_pairs = node_idx_pairs_no

    # print('PROBLEM NODES:', problem['nodes'])
    # print('NODE IDX PAIRS:', node_idx_pairs)

    vol_frac = vox_space(problem, node_idx_pairs, resolution=VOLFRAC_RESOLUTION)

    if normalize is True:
        set_norms(problem)
        norms = problem['volfrac_norms']
        v_min, v_max = norms[0], norms[1]
        vol_frac_norm = (vol_frac - v_min) / (v_max - v_min)
        if vol_frac_norm < 0.0:
            vol_frac_norm = 0.0
        vol_frac = vol_frac_norm

    return vol_frac


if __name__ == '__main__':

    num_bits = rep.get_num_bits(val_problem)
    design_rep =[1 for x in range(num_bits)]
    print(rep.get_design_metrics(val_problem, design_rep))
    rep.viz(val_problem, design_rep, 'curr_problem.png')

























