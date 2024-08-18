from abc import ABC, abstractmethod
import numpy as np
import random
import math
import pint
from copy import deepcopy
import json





class AbstractProblem(ABC):

    def __init__(self):
        self.ureg = pint.UnitRegistry()


    @staticmethod
    def enumerate(params, p_type='type1', **kwargs):
        pass







    # -------------------------------------------------------
    # Helpers
    # -------------------------------------------------------

    @staticmethod
    def get_mesh(x_range, x_res, y_range, y_res):
        x = np.linspace(0, x_range, x_res)
        y = np.linspace(0, y_range, y_res)
        nodes = []
        for i in x:
            for j in y:
                nodes.append([i, j])
        nodes = sorted(nodes, key=lambda x: (x[0], x[1]))
        return nodes

    @staticmethod
    def drop_nodes_enum(problem, dropout=0.1, ss=10):
        # Returns a set of copies of the problem with some nodes dropped
        probs_id = set()
        probs = []
        count = 0
        while count < ss:
            p = AbstractProblem.drop_nodes(problem, dropout=dropout)
            nn = len(p['nodes'])
            p_str = json.dumps(p['nodes'])
            if p_str not in probs_id and nn > 4:
                probs_id.add(p_str)
                probs.append(p)
            count += 1
        return probs

    @staticmethod
    def drop_nodes(problem, dropout=0.1):
        # Returns a copy of the problem with some nodes dropped
        from combench.models.truss.representation import get_edge_nodes, get_load_nodes, get_all_fixed_nodes
        load_cond = problem['load_conds'][0]
        p = deepcopy(problem)
        lc = deepcopy(load_cond)
        n_nodes = len(p['nodes'])
        edge_nodes = get_edge_nodes(p)
        load_nodes = get_load_nodes(lc)
        fixed_nodes = get_all_fixed_nodes(p)
        other_nodes = [x for x in range(n_nodes) if x not in (edge_nodes + load_nodes + fixed_nodes)]
        droppable_nodes = []
        droppable_nodes.extend(other_nodes)
        if len(fixed_nodes) > 2:
            droppable_nodes.extend(fixed_nodes[:-2])
        non_load_edge_nodes = [x for x in edge_nodes if x not in (load_nodes + fixed_nodes)]
        droppable_nodes.extend(non_load_edge_nodes)
        droppable_nodes.sort(reverse=True)
        for drop_node in droppable_nodes:
            if random.random() < dropout:
                del p['nodes'][drop_node]
                del p['nodes_dof'][drop_node]
                del lc[drop_node]
        p['load_conds'] = [lc]
        return p


    @staticmethod
    def clear_loads(problem):
        for idx1, load_cond in enumerate(problem['load_conds']):
            for idx2, n_load in enumerate(load_cond):
                problem['load_conds'][idx1][idx2] = [0, 0]
        return problem


























