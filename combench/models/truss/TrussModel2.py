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


NUM_PROCS = 4


class TrussModel2(Model):

    def __init__(self, problem_formulation):
        super().__init__(problem_formulation)
        self.norms = self.load_norms()
        print('Norm values: {}'.format(self.norms))
        self.procs = []

    def init_procs(self):
        self.num_procs = NUM_PROCS
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
        if 'norms' in self.problem_store:
            return self.problem_store['norms']

        norms = truss.set_norms(self.problem_formulation)
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

    def evaluate_batch2(self, designs, normalize=True, pool=None):
        if pool is None:
            with concurrent.futures.ProcessPoolExecutor() as ppool:
                evals = list(ppool.map(self.evaluate, designs))
        else:
            params = [(self.problem_formulation, design) for design in designs]
            evals = list(pool.map(local_eval, params))
        return evals

    def evaluate_batch(self, designs, normalize=True):
        if len(self.procs) == 0:
            # print('INITIALIZING PROCS')
            self.init_procs()
            # print('PROCS INITIALIZED', len(self.procs))

        # Decompose designs into chunks
        chunk_size = math.ceil(len(designs) / self.num_procs)
        chunks = [designs[i:i + chunk_size] for i in range(0, len(designs), chunk_size)]

        r_queues = []
        for i, chunk in enumerate(chunks):
            self.procs[i][1].put((self.problem_formulation, chunk))
            r_queues.append(self.procs[i][2])

        # print('Waiting for results')
        evals = []
        for r_queue in r_queues:
            evals += r_queue.get()
        # print('Results received')
        return evals

    def evaluate_batch_async(self, designs, normalize=True):
        if len(self.procs) == 0:
            self.init_procs()

        # Decompose designs into chunks
        chunk_size = math.ceil(len(designs) / self.num_procs)
        chunks = [designs[i:i + chunk_size] for i in range(0, len(designs), chunk_size)]

        r_queues = []
        for i, chunk in enumerate(chunks):
            self.procs[i][1].put((self.problem_formulation, chunk))
            r_queues.append(self.procs[i][2])

        return r_queues

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
                # node[0], node[1], node_dof[0], node_dof[1], node_load[0], node_load[1]
                node_load[0], node_load[1]
            ])
            node_vectors.append(node_vector)
        return node_vectors



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
    truss_model = TrussModel2(problem)


    # design_str = '111010000010100010100000101010'
    # design_array = [int(bit) for bit in design_str]
    design_array = truss_model.random_design()
    design_array = [1 for x in design_array]
    curr_time = time.time()
    objectives = truss_model.evaluate(design_array)
    print('Objectives:', objectives)
    print("Time taken: ", time.time() - curr_time)
