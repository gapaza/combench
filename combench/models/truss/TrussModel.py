import numpy as np
import string
import random
import config
import time
import math

from combench.nn.trussDecoderUMD2 import model_embed_dim
reduced_embed_dim = model_embed_dim - 0


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


NUM_PROCS = 16


class TrussModel(Model):

    def __init__(self, problem_formulation, num_procs=1):
        super().__init__(problem_formulation)
        self.norms = self.load_norms()
        print('Norm values: {}'.format(self.norms))
        self.procs = []
        self.num_procs = num_procs

        # Normalization Information
        self.coord_max = max([max(node) for node in self.problem_formulation['nodes']])
        self.coord_min = min([min(node) for node in self.problem_formulation['nodes']])
        self.load_max = max([max(load) for load in self.problem_formulation['load_conds'][0]])
        self.load_min = min([min(load) for load in self.problem_formulation['load_conds'][0]])

        self.norm_nodes = self.normalize_components(self.problem_formulation['nodes'], self.coord_min, self.coord_max)
        self.norm_loads = self.normalize_components(self.problem_formulation['load_conds'][0], self.load_min, self.load_max)


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
        if random.random() < 0.5:
            return truss.rep.random_sample_1(self.problem_formulation)
        else:
            return truss.rep.random_sample_2(self.problem_formulation)

    def evaluate(self, design, normalize=True):
        stiff_vals = truss.eval_stiffness(self.problem_formulation, design, normalize=normalize)
        # print('RETURNED STIFF VAL', stiff_vals)
        stiff = stiff_vals[0] * -1.0  # maximize stiffness_old
        if stiff == 0:
            volfrac = 1
        else:
            volfrac = truss.eval_volfrac(self.problem_formulation, design, normalize=normalize)
        return stiff, volfrac

    # ---------------------------------------
    # Problem Encoding
    # ---------------------------------------


    def get_encoding(self, rand=False):
        # Create a vector for each node with the following values:
        # [x, y, dofx, dofy, fx, fy]
        problem = self.problem_formulation
        load_cond = problem['load_conds'][0]
        node_vectors = []
        for idx, node in enumerate(problem['nodes']):
            node_dof = problem['nodes_dof'][idx]
            node_load = load_cond[idx]

            # node_list = [node_load[0], node_load[1]]
            node_list = [node_dof[0], node_dof[1], node_load[0], node_load[1]]
            # node_list = [node[0], node[1], node_dof[0], node_dof[1], node_load[0], node_load[1]]

            node_vector = np.array(self.fill_embedding(node_list))
            node_vectors.append(node_vector)
        # if rand is True:
        #     random.shuffle(node_vectors)
        return node_vectors

    def fill_embedding(self, node_list, embedding_dim=reduced_embed_dim):
        embedding = []
        while len(embedding) < embedding_dim:
            embedding += node_list
        if len(embedding) > embedding_dim:
            embedding = embedding[:embedding_dim]
        return embedding


    def get_padded_encoding(self, pad_len, rand=False):
        encoding = self.get_encoding(rand=rand)
        padding_mask = [1 for x in encoding]  # 1s where actual nodes are
        padding_mask += [0 for x in range(pad_len - len(encoding))]
        encoding += [[0 for x in range(reduced_embed_dim)] for x in range(pad_len - len(encoding))]
        return encoding, padding_mask


    def get_dynamic_encoding(self, number_of_neurons, pad_len):
        # Create a vector for each node with the following values:
        # [x, y, dofx, dofy, fx, fy]
        problem = self.problem_formulation
        load_cond_norm = self.norm_loads
        node_pos_norm = self.norm_nodes

        # Map neuron index to nodes
        neuron_indices = [x for x in range(number_of_neurons)]
        rand_neuron_indices = random.sample(neuron_indices, len(problem['nodes']))
        neuron_map = {}
        for idx, neuron_idx in enumerate(rand_neuron_indices):
            neuron_map[neuron_idx] = idx
        # rand_neuron_indices = [-1] + rand_neuron_indices  # Add 1 to start for start token (becomes 1 after shift)

        # Assemble node vectors
        node_vectors = []
        for idx, node in enumerate(problem['nodes']):
            node_dof = problem['nodes_dof'][idx]
            node_load = load_cond_norm[idx]
            node_list = [node_pos_norm[idx][0], node_pos_norm[idx][1], node_dof[0], node_dof[1], node_load[0], node_load[1]]
            node_vector = np.array(self.fill_embedding(node_list))
            node_vectors.append(node_vector)

        # Pad the encoding
        encoding = node_vectors
        encoding += [[0 for x in range(reduced_embed_dim)] for x in range(pad_len - len(encoding))]

        # Create padding mask
        padding_mask = [1 for x in encoding]  # 1s where actual nodes are
        padding_mask += [0 for x in range(pad_len - len(encoding))]

        rand_neuron_indices_shifted = [x + 2 for x in rand_neuron_indices]  # Shift by 1 for input vocabulary
        rand_neuron_indices_padded = rand_neuron_indices_shifted + [0 for x in range(pad_len - len(rand_neuron_indices_shifted))]
        rand_neuron_indices_padded = np.array(rand_neuron_indices_padded)

        return encoding, padding_mask, neuron_map, rand_neuron_indices_padded




    def normalize_components(self, items, min_val, max_val):
        norm_items = []
        for values in items:
            normalized_values = [(value - min_val) / (max_val - min_val) for value in values]
            norm_items.append(normalized_values)
        return norm_items





if __name__ == '__main__':
    from combench.models.truss import train_problems, val_problems

    param_dict = {
        'x_range': 3,
        'y_range': 3,
        'x_res': 3,
        'y_res': 3,
        'radii': 0.2,
        'y_modulus': 210e9
    }
    problem = val_problems[0]
    truss_model = TrussModel(problem)


    # design_str = '111010000010100010100000101010'
    # design_array = [int(bit) for bit in design_str]
    design_array = truss_model.random_design()
    design_array = [1 for x in design_array]
    curr_time = time.time()
    objectives = truss_model.evaluate(design_array)
    print('Objectives:', objectives)
    print("Time taken: ", time.time() - curr_time)
