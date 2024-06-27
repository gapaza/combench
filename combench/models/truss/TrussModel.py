import numpy as np
import string
import random
import config
import time
import math


# from combench.models.truss.vol.TrussVolumeFraction import TrussVolumeFraction
# from combench.models.truss.stiffness.TrussStiffness import TrussStiffness
# from combench.models.truss import sidenum_nvar_map
from combench.core.model import Model
from combench.models.utils import random_binary_design
from combench.models import truss
import concurrent.futures
from combench.models.truss.eval_process import EvaluationProcess
import multiprocessing
# Update to fork instead of spawn for multiprocessing


def local_eval(params):
    problem, design_rep = params
    stiff_vals = truss.eval_stiffness(problem, design_rep, normalize=True)
    stiff = stiff_vals[0] * -1.0  # maximize stiffness_old
    volfrac = truss.eval_volfrac(problem, design_rep)
    return stiff, volfrac


NUM_PROCS = 1


class TrussModel(Model):

    def __init__(self, problem_formulation, num_procs=1):
        super().__init__(problem_formulation)
        self.norms = self.load_norms()
        print('Norm values: {}'.format(self.norms))
        self.procs = []
        self.num_procs = num_procs

    def init_procs(self):
        # self.num_procs = NUM_PROCS
        self.procs = []
        for x in range(self.num_procs):
            request_queue = multiprocessing.Queue()
            response_queue = multiprocessing.Queue()
            eval_proc = EvaluationProcess(request_queue, response_queue)
            eval_proc.start()
            self.procs.append([
                eval_proc, request_queue, response_queue
            ])

    def close_procs(self):
        for proc in self.procs:
            proc[0].terminate()

    def __del__(self):
        self.close_procs()


    def load_norms(self):
        if 'norms' in self.problem_formulation:
            return self.problem_formulation['norms']
        if 'norms' in self.problem_store:
            return self.problem_store['norms']

        truss.set_norms(self.problem_formulation)
        norms = self.problem_formulation['norms']

        self.problem_store['norms'] = norms
        self.save_problem_store()
        return norms

    def random_design(self):
        return truss.rep.random_sample_1(self.problem_formulation)

    def evaluate(self, design, normalize=True):
        stiff_vals = truss.eval_stiffness(self.problem_formulation, design, normalize=True)
        stiff = stiff_vals[0] * -1.0  # maximize stiffness_old
        volfrac = truss.eval_volfrac(self.problem_formulation, design)
        return stiff, volfrac

    def get_encoding(self):
        # Create a vector for each node with the following values:
        # [x, y, dofx, dofy, fx, fy]
        problem = self.problem_formulation
        load_cond = problem['load_conds'][0]
        node_vectors = []
        for idx, node in enumerate(problem['nodes']):
            node_dof = problem['nodes_dof'][idx]
            node_load = load_cond[idx]
            node_vector = np.array([
                node[0], node[1], node_dof[0], node_dof[1], node_load[0], node_load[1]
                # node_load[0], node_load[1]
            ])
            node_vectors.append(node_vector)
        return node_vectors

    def get_padded_encoding(self, pad_len):
        encoding = self.get_encoding()
        padding_mask = [1 for x in encoding]  # 1s where actual nodes are
        padding_mask += [0 for x in range(pad_len - len(encoding))]
        encoding += [[0 for x in range(6)] for x in range(pad_len - len(encoding))]
        return encoding, padding_mask


from combench.models.truss import Cantilever

if __name__ == '__main__':
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
    truss_model = TrussModel(problem)


    # design_str = '111010000010100010100000101010'
    # design_array = [int(bit) for bit in design_str]
    design_array = truss_model.random_design()
    design_array = [1 for x in design_array]
    curr_time = time.time()
    objectives = truss_model.evaluate(design_array)
    print('Objectives:', objectives)
    print("Time taken: ", time.time() - curr_time)
