import config
import json
import math
from collections import deque
import numpy as np
import random
from copy import deepcopy
import os
import pandas as pd
import matplotlib.pyplot as plt
import time
from typing import List, Tuple, Dict, Optional, Set, Any


from combench.ga.NSGA2 import NSGA2
from combench.core.design import Design
from combench.ga.UnconstrainedPop import UnconstrainedPop
from combench.ga.ConstrainedPop import ConstrainedPop
from combench.models import truss
from combench.models.truss.eval_process import EvaluationProcessManager
from combench.models.truss import plotting
from combench.models.truss.TrussModel import TrussModel


#       _____            _
#      |  __ \          (_)
#      | |  | | ___  ___ _  __ _ _ __
#      | |  | |/ _ \/ __| |/ _` | '_ \
#      | |__| |  __/\__ \ | (_| | | | |
#      |_____/ \___||___/_|\__, |_| |_|
#                           __/ |
#                          |___/


class TrussDesign(Design):
    def __init__(self, vector, problem):
        super().__init__(vector, problem)

        # This is the problem object holding nodes, nodes_dof, etc...
        self.problem_formulation = problem.problem_formulation

    def random_design(self):
        return self.problem.random_design()

    def mutate(self):
        prob_mutate = 1.0 / self.num_vars
        for i in range(self.num_vars):
            if random.random() < prob_mutate:
                if self.vector[i] == 0:
                    self.vector[i] = 1
                else:
                    self.vector[i] = 0

    def evaluate(self):
        if self.is_evaluated() is True:
            return self.objectives
        stiff, vol_frac = self.problem.evaluate(self.vector)
        self.objectives = [stiff, vol_frac]

        constraint_score = self.problem.evaluate_constraints(self.vector)
        if constraint_score > 0:
            self.is_feasible = False

        self.feasibility_score = constraint_score
        if self.feasibility_score > 0:
            self.is_feasible = False
        else:
            self.is_feasible = True

        return self.objectives

    def get_plotting_objectives(self):
        return [-self.objectives[0], self.objectives[1]]

#       _____                  _       _   _
#      |  __ \                | |     | | (_)
#      | |__) |__  _ __  _   _| | __ _| |_ _  ___  _ __
#      |  ___/ _ \| '_ \| | | | |/ _` | __| |/ _ \| '_ \
#      | |  | (_) | |_) | |_| | | (_| | |_| | (_) | | | |
#      |_|   \___/| .__/ \__,_|_|\__,_|\__|_|\___/|_| |_|
#                 | |
#                 |_|


class TrussPopulation(ConstrainedPop):

    def __init__(self, pop_size, ref_point, problem):
        super().__init__(pop_size, ref_point, problem)
        self.eval_manager = None
        self.unique_designs_epoch = []
        self.epoch = 0

    def init_eval_mamanger(self):
        if self.eval_manager is None:
            print('INITIALIZING EVAL MANAGER --> CREATING', 24, 'PROCESSES')
            self.eval_manager = EvaluationProcessManager(num_processes=24)

    def create_design(self, vector=None):
        design = TrussDesign(vector, self.problem)
        return design

    def eval_population(self):
        self.epoch += 1

        # Evaluate unknown designs
        unkonwn_designs = [design for design in self.designs if not design.is_evaluated()]
        unknown_designs_vectors = [design.vector for design in unkonwn_designs]
        if len(unknown_designs_vectors) > 0:
            self.init_eval_mamanger()
            batch_probs = [self.problem.problem_formulation for _ in range(len(unknown_designs_vectors))]
            # unknown_designs_objectives = self.eval_manager.evaluate_store(batch_probs, unknown_designs_vectors)
            unknown_designs_objectives = self.eval_manager.evaluate(batch_probs, unknown_designs_vectors)
            for design, objs in zip(unkonwn_designs, unknown_designs_objectives):
                design.objectives = [objs[0], objs[1]]
                design.feasibility_score = objs[2]
                # design.feasibility_score = 0.0
                if objs[2] > 0:
                    design.is_feasible = False
                else:
                    design.is_feasible = True

                # design.objectives = objs
                # design.is_feasible = True
                design.epoch = deepcopy(self.epoch)

        # Collect objectives
        objectives = []
        for design in self.designs:
            objs = design.evaluate()
            design_str = design.get_design_str()
            if design_str not in self.unique_designs_bitstr:
                self.unique_designs_bitstr.add(design_str)
                self.unique_designs.append(design)
                self.nfe += 1
                self.unique_designs_epoch.append(deepcopy(self.epoch))
            objectives.append(objs)
        return objectives

    def plot_population(self, save_dir):
        p = self.problem.problem_formulation
        pareto_plot_file = os.path.join(save_dir, 'designs_pareto.png')
        pareto_json_file = os.path.join(save_dir, 'designs_pareto.json')
        all_plot_file = os.path.join(save_dir, 'designs_all.png')
        all_epoch_plot_file = os.path.join(save_dir, 'epochs_all.png')
        all_weights_file = os.path.join(save_dir, 'weights_all.png')

        # Pareto designs
        if len(self.designs) > 0:
            plotting.plot_pareto_designs(self.designs, pareto_plot_file, pareto_json_file)

        # Viz 3 Individual Select Pareto Designs
        if len(self.designs) > 5:
            pareto_dir = os.path.join(save_dir, 'pareto')
            if not os.path.exists(pareto_dir):
                os.makedirs(pareto_dir)
            else:
                for file in os.listdir(pareto_dir):
                    os.remove(os.path.join(pareto_dir, file))
            plotting.plot_select_designs(p, self.designs, pareto_dir)

        # Feasible select designs
        if len(self.designs) > 3:
            feasible_dir = os.path.join(save_dir, 'feasible')
            if not os.path.exists(feasible_dir):
                os.makedirs(feasible_dir)
            else:
                for file in os.listdir(feasible_dir):
                    os.remove(os.path.join(feasible_dir, file))
            plotting.plot_feasible_designs(p, self.designs, feasible_dir)

        # All designs
        if len(self.unique_designs) > 0:
            plotting.plot_all_designs(self.unique_designs, all_plot_file)

        # Plot weight graph
        if len(self.unique_designs) > 0:
            plotting.plot_weight_graph(self.unique_designs, all_weights_file)

ALGORITHM_TIMEOUT_SECONDS = 120  # 1 minute


# -----------------------------------------------------------
# SHINKA ALGORITHM
# -----------------------------------------------------------
# EVOLVE-BLOCK-START
import random
from itertools import chain, combinations

def search_algorithm(truss_problem):
    """Algorithm for finding the Pareto front of a 2D truss design problem.
    This method must save Pareto front designs using the truss_problem.save_solutions(np.ndarray) method.

    Args:
        truss_problem: truss problem to be evaluated.
    Saves:
        designs: np.array encoding the designs that maximize the Pareto front hypervolume and satisfy the constraints.
    """
    # Helpers: Pareto dominance and archive insert
    def dominates(a, b):
        # Both objectives are minimized here: f1 = -stiffness, f2 = volume fraction
        return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])

    def pareto_insert(design_bits, fvals, archive_objs, archive_bits):
        # Reject if dominated by any existing
        for fo in archive_objs:
            if dominates(fo, fvals):
                return False
        # Remove any that are dominated by the newcomer
        keep_objs = []
        keep_bits = []
        for fo, db in zip(archive_objs, archive_bits):
            if not dominates(fvals, fo):
                keep_objs.append(fo)
                keep_bits.append(db)
        keep_objs.append(fvals)
        keep_bits.append(design_bits)
        archive_objs[:] = keep_objs
        archive_bits[:] = keep_bits
        return True

    # Lightweight mutation + repair to enforce overlap constraint
    def mutate_repair(base_bits, n_bits, truss_problem, k_flips=0, max_repair_steps=40, max_tries=2):
        # Conflict-aware mutate+repair with bias toward promising edges and cached geometry.
        if k_flips <= 0:
            k_flips = max(1, int(0.01 * n_bits))

        # One-time geometry/conflict cache
        state = getattr(mutate_repair, "_state", None)
        if state is None:
            import math
            pf = truss_problem.get_problem_formulation()
            nodes_local = pf['nodes']
            bit_to_pair_local = []
            coords_by_bit_local = []
            for i in range(len(nodes_local)):
                for j in range(i + 1, len(nodes_local)):
                    bit_to_pair_local.append((i, j))
                    coords_by_bit_local.append((nodes_local[i], nodes_local[j]))
            m = len(bit_to_pair_local)
            lengths = [math.hypot(a[0]-b[0], a[1]-b[1]) for (a, b) in coords_by_bit_local]

            def orient(a, b, c):
                return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

            def on_segment(a, b, p):
                return (min(a[0], b[0]) - 1e-9 <= p[0] <= max(a[0], b[0]) + 1e-9 and
                        min(a[1], b[1]) - 1e-9 <= p[1] <= max(a[1], b[1]) + 1e-9)

            def segments_intersect(a, b, c, d):
                if a == c or a == d or b == c or b == d:
                    return False
                if max(a[0], b[0]) + 1e-9 < min(c[0], d[0]) - 1e-9 or max(c[0], d[0]) + 1e-9 < min(a[0], b[0]) - 1e-9:
                    return False
                if max(a[1], b[1]) + 1e-9 < min(c[1], d[1]) - 1e-9 or max(c[1], d[1]) + 1e-9 < min(a[1], b[1]) - 1e-9:
                    return False
                o1 = orient(a,b,c); o2 = orient(a,b,d); o3 = orient(c,d,a); o4 = orient(c,d,b)
                if abs(o1) < 1e-12 and on_segment(a,b,c): return True
                if abs(o2) < 1e-12 and on_segment(a,b,d): return True
                if abs(o3) < 1e-12 and on_segment(c,d,a): return True
                if abs(o4) < 1e-12 and on_segment(c,d,b): return True
                if (o1>0 and o2<0 or o1<0 and o2>0) and (o3>0 and o4<0 or o3<0 and o4>0):
                    return True
                return False

            conflict_sets_local = [set() for _ in range(m)]
            for i in range(m):
                a, b = coords_by_bit_local[i]
                for j in range(i+1, m):
                    c, d = coords_by_bit_local[j]
                    if segments_intersect(a, b, c, d):
                        conflict_sets_local[i].add(j)
                        conflict_sets_local[j].add(i)

            state = {
                'lengths': lengths,
                'conflict_sets': conflict_sets_local,
                'm': m
            }
            mutate_repair._state = state

        lengths = state['lengths']
        conflict_sets_local = state['conflict_sets']

        for _ in range(max_tries):
            bits = list(base_bits)

            zeros = [i for i, v in enumerate(bits) if v == 0]
            ones = [i for i, v in enumerate(bits) if v == 1]

            add_k = min(len(zeros), max(1, int(0.5 * k_flips)))
            rem_k = min(len(ones), max(0, k_flips - add_k))

            chosen = set()

            # Preferential additions (high edge_score/base_weight)
            if zeros and add_k > 0:
                try:
                    w = np.array([edge_score[i] + 0.4 * base_weight[i] + 1e-8 for i in zeros], dtype=float)
                    s = w.sum()
                    if s <= 0:
                        picks = random.sample(zeros, add_k)
                    else:
                        probs = w / s
                        picks = list(np.random.choice(zeros, size=add_k, replace=False, p=probs))
                except Exception:
                    picks = random.sample(zeros, add_k)
                for idx in picks:
                    chosen.add(idx)
                    bits[idx] = 1

            # Preferential removals (low-scoring, easier-to-remove)
            if ones and rem_k > 0:
                try:
                    w_rem = np.array([edge_score[i] + 0.4 * base_weight[i] + 1e-8 for i in ones], dtype=float)
                    inv = 1.0 / (w_rem + 1e-9)
                    s2 = inv.sum()
                    if s2 <= 0:
                        picks_rem = random.sample(ones, rem_k)
                    else:
                        probs_rem = inv / s2
                        picks_rem = list(np.random.choice(ones, size=rem_k, replace=False, p=probs_rem))
                except Exception:
                    picks_rem = random.sample(ones, rem_k)
                for idx in picks_rem:
                    chosen.add(idx)
                    bits[idx] = 0

            # Fill remaining flips randomly if needed
            current_flips = len(chosen)
            if current_flips < min(k_flips, n_bits):
                remaining = [i for i in range(n_bits) if i not in chosen]
                need = min(min(k_flips, n_bits) - current_flips, len(remaining))
                if need > 0:
                    for idx in random.sample(remaining, need):
                        bits[idx] = 1 - bits[idx]

            # Fast repair with conflict-aware removals
            steps = 0
            while steps < max_repair_steps:
                present = [i for i, v in enumerate(bits) if v]
                present_set = set(present)
                overlaps = []
                for i in present:
                    for j in conflict_sets_local[i]:
                        if j in present_set and i < j:
                            overlaps.append((i, j))
                if not overlaps:
                    break
                conflict_count = {}
                for a, b in overlaps:
                    conflict_count[a] = conflict_count.get(a, 0) + 1
                    conflict_count[b] = conflict_count.get(b, 0) + 1
                best = None
                best_key = None
                for idx, cnt in conflict_count.items():
                    key = (cnt, lengths[idx], -edge_score[idx])
                    if best_key is None or key > best_key:
                        best_key = key
                        best = idx
                if best is None:
                    if present:
                        best = random.choice(present)
                    else:
                        break
                bits[best] = 0
                steps += 1

            # Fallback trimming with API check
            if truss_problem.evaluate_constraints(bits) != 0:
                ones2 = [i for i, v in enumerate(bits) if v]
                ones2.sort(key=lambda x: lengths[x], reverse=True)
                for idx_rm in ones2:
                    bits[idx_rm] = 0
                    if truss_problem.evaluate_constraints(bits) == 0:
                        break

            if truss_problem.evaluate_constraints(bits) == 0:
                return bits
        return None

    # Get the problem formulation
    problem_formulation = truss_problem.get_problem_formulation()
    nodes = problem_formulation['nodes']

    # Get fixed and loaded node indices
    all_node_idx = [x for x in range(len(nodes))]
    fixed_node_idx = truss_problem.get_fixed_nodes()
    load_node_idx = truss_problem.get_load_nodes()

    # Global problem size and mappings
    n_bits = truss_problem.get_n_bits()
    import math
    # Precompute bit -> (i,j), coordinates, lengths, base weights
    bit_to_pair = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            bit_to_pair.append((i, j))
    assert len(bit_to_pair) == n_bits
    # fast lookup from pair to bit index
    pair_to_bit = {pair: idx for idx, pair in enumerate(bit_to_pair)}
    coords_by_bit = [(nodes[i], nodes[j]) for (i, j) in bit_to_pair]
    edge_len = [math.hypot(a[0] - b[0], a[1] - b[1]) for (a, b) in coords_by_bit]
    base_weight = []
    for idx2, (ii, jj) in enumerate(bit_to_pair):
        w = 1.0 / (edge_len[idx2] + 1e-9)
        if ii in load_node_idx or jj in load_node_idx:
            w += 2.5
        if ii in fixed_node_idx or jj in fixed_node_idx:
            w += 1.2
        base_weight.append(w)
    fixed_fixed_bits = set(idx for idx, (i, j) in enumerate(bit_to_pair) if i in fixed_node_idx and j in fixed_node_idx)

    # Robust intersection helpers and conflict sets for planarity checks
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    def on_segment(a, b, p):
        return (min(a[0], b[0]) - 1e-9 <= p[0] <= max(a[0], b[0]) + 1e-9 and
                min(a[1], b[1]) - 1e-9 <= p[1] <= max(a[1], b[1]) + 1e-9)
    def segments_intersect(a, b, c, d):
        # Allow shared endpoints (not considered overlap)
        if a == c or a == d or b == c or b == d:
            return False
        # Quick bbox reject
        if max(a[0], b[0]) + 1e-9 < min(c[0], d[0]) - 1e-9 or max(c[0], d[0]) + 1e-9 < min(a[0], b[0]) - 1e-9:
            return False
        if max(a[1], b[1]) + 1e-9 < min(c[1], d[1]) - 1e-9 or max(c[1], d[1]) + 1e-9 < min(a[1], b[1]) - 1e-9:
            return False
        o1 = orient(a, b, c); o2 = orient(a, b, d)
        o3 = orient(c, d, a); o4 = orient(c, d, b)
        if abs(o1) < 1e-12 and on_segment(a, b, c): return True
        if abs(o2) < 1e-12 and on_segment(a, b, d): return True
        if abs(o3) < 1e-12 and on_segment(c, d, a): return True
        if abs(o4) < 1e-12 and on_segment(c, d, b): return True
        if (o1 > 0 and o2 < 0 or o1 < 0 and o2 > 0) and (o3 > 0 and o4 < 0 or o3 < 0 and o4 > 0):
            return True
        return False

    conflict_sets = [set() for _ in range(n_bits)]
    for a_idx in range(n_bits):
        a0, a1 = coords_by_bit[a_idx]
        for b_idx in range(a_idx + 1, n_bits):
            b0, b1 = coords_by_bit[b_idx]
            if segments_intersect(a0, a1, b0, b1):
                conflict_sets[a_idx].add(b_idx)
                conflict_sets[b_idx].add(a_idx)

    # Connectivity helpers: ensure load nodes connect to fixed nodes with minimal, non-conflicting edges
    def compute_components(bitlist):
        adj = [[] for _ in range(len(nodes))]
        for e_idx, val in enumerate(bitlist):
            if not val:
                continue
            u, v = bit_to_pair[e_idx]
            adj[u].append(v)
            adj[v].append(u)
        comp_id = [-1] * len(nodes)
        comps = {}
        cid = 0
        for s in range(len(nodes)):
            if comp_id[s] != -1:
                continue
            stack = [s]
            comp_id[s] = cid
            comps[cid] = [s]
            while stack:
                x = stack.pop()
                for y in adj[x]:
                    if comp_id[y] == -1:
                        comp_id[y] = cid
                        comps[cid].append(y)
                        stack.append(y)
            cid += 1
        return comp_id, comps

    def loads_connected_to_fixed(comp_id):
        fixed_comps = set(comp_id[f] for f in fixed_node_idx) if fixed_node_idx else set()
        for ln in load_node_idx:
            if comp_id[ln] not in fixed_comps:
                return False
        return True

    def ensure_connectivity(bitlist, max_add_edges=3):
        bl = bitlist[:]
        comp_id, comps = compute_components(bl)
        if loads_connected_to_fixed(comp_id):
            return bl

        fixed_comps = set(comp_id[f] for f in fixed_node_idx) if fixed_node_idx else set()
        anchor_nodes = set()
        for cid, ns in comps.items():
            if cid in fixed_comps:
                anchor_nodes.update(ns)
        load_only_cids = set()
        for ln in load_node_idx:
            if comp_id[ln] not in fixed_comps:
                load_only_cids.add(comp_id[ln])
        target_nodes = set()
        for cid in load_only_cids:
            target_nodes.update(comps.get(cid, []))
        if not anchor_nodes or not target_nodes:
            return bl

        def min_dist_to_set(node_idx, others):
            x, y = nodes[node_idx]
            best = 1e18
            for o in others:
                xo, yo = nodes[o]
                d = (x - xo) * (x - xo) + (y - yo) * (y - yo)
                if d < best:
                    best = d
            return best

        anchor_list = sorted(list(anchor_nodes), key=lambda a: min_dist_to_set(a, set(load_node_idx)))[:min(8, len(anchor_nodes))]
        target_list = sorted(list(target_nodes), key=lambda t: min_dist_to_set(t, set(fixed_node_idx)))[:min(8, len(target_nodes))]

        present = set(i for i, v in enumerate(bl) if v)
        added = 0
        while added < max_add_edges:
            candidates = []
            for u in target_list:
                for v in anchor_list:
                    if u == v:
                        continue
                    a, b = (u, v) if u < v else (v, u)
                    idx = pair_to_bit.get((a, b))
                    if idx is None or idx in fixed_fixed_bits:
                        continue
                    if bl[idx] == 1:
                        continue
                    # conflict check
                    conflict = False
                    for ex in conflict_sets[idx]:
                        if bl[ex] == 1:
                            conflict = True; break
                    if conflict:
                        continue
                    score = base_weight[idx] + 0.2 / (edge_len[idx] + 1e-9)
                    candidates.append((score, idx))
            if not candidates:
                break
            candidates.sort(key=lambda t: -t[0])
            _, best_idx = candidates[0]
            bl[best_idx] = 1
            present.add(best_idx)
            added += 1
            comp_id, comps = compute_components(bl)
            if loads_connected_to_fixed(comp_id):
                break

        if conflict_density(bl) > 0.0:
            bl = repair_planarity(bl)
        return bl

    # Two-hop neighborhood extraction around a design to constrain mutations
    def get_two_hop_allowed_indices(bits, depth=2, max_nodes=40):
        present_edges = [i for i, v in enumerate(bits) if v]
        if not present_edges:
            return [i for i in range(n_bits) if i not in fixed_fixed_bits]
        adj = [[] for _ in range(len(nodes))]
        used_nodes = set()
        for e_idx in present_edges:
            u, v = bit_to_pair[e_idx]
            adj[u].append(v); adj[v].append(u)
            used_nodes.add(u); used_nodes.add(v)
        near = set(used_nodes)
        frontier = list(used_nodes)
        for _ in range(max(1, depth)):
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if v not in near:
                        near.add(v); nxt.append(v)
            frontier = nxt
            if not frontier:
                break
        if len(near) > max_nodes:
            anchors = set(load_node_idx + fixed_node_idx)
            def mindist(i):
                x, y = nodes[i]
                best = 1e18
                for a in anchors:
                    xa, ya = nodes[a]
                    d = (x-xa)*(x-xa) + (y-ya)*(y-ya)
                    if d < best: best = d
                return best
            near = set(sorted(list(near), key=mindist)[:max_nodes])
        allowed = []
        for idx, (u, v) in enumerate(bit_to_pair):
            if idx in fixed_fixed_bits:
                continue
            if u in near or v in near:
                allowed.append(idx)
        return allowed

    # Leader selection helper: crowding distance priority
    def compute_crowding_priority(objs):
        n = len(objs)
        if n == 0:
            return []
        idxs = list(range(n))
        dist = [0.0] * n
        for d in [0, 1]:
            order = sorted(idxs, key=lambda k: objs[k][d])
            dist[order[0]] = dist[order[-1]] = float('inf')
            fmin = objs[order[0]][d]; fmax = objs[order[-1]][d]
            denom = (fmax - fmin) if fmax > fmin else 1.0
            for i in range(1, n - 1):
                prev = objs[order[i - 1]][d]
                nxt = objs[order[i + 1]][d]
                dist[order[i]] += (nxt - prev) / denom
        return dist

    # Subset-restricted mutation and planarity repair
    def mutate_subset_repair(base_bits, allowed_idx, n_bits, truss_problem, k_flips=0, max_repair_steps=30, max_tries=2):
        if not allowed_idx:
            return None
        if k_flips <= 0:
            k_flips = max(1, int(0.008 * n_bits))
        state = getattr(mutate_repair, "_state", None)
        if state is None:
            tmp = mutate_repair(base_bits, n_bits, truss_problem, k_flips=1, max_repair_steps=1, max_tries=1)
            state = getattr(mutate_repair, "_state", None)
            if state is None:
                return None
        lengths = state['lengths']
        conflict_sets_local = state['conflict_sets']
        allowed = list(allowed_idx)
        for _ in range(max_tries):
            bits = list(base_bits)
            cand_zero = [i for i in allowed if bits[i] == 0]
            cand_one = [i for i in allowed if bits[i] == 1]
            add_k = min(len(cand_zero), max(1, int(0.6 * k_flips)))
            rem_k = min(len(cand_one), max(0, k_flips - add_k))
            if add_k > 0 and cand_zero:
                picks = random.sample(cand_zero, add_k)
                for i in picks:
                    bits[i] = 1
            if rem_k > 0 and cand_one:
                picks = random.sample(cand_one, rem_k)
                for i in picks:
                    bits[i] = 0
            # conflict repair
            steps = 0
            while steps < max_repair_steps:
                present = [i for i, v in enumerate(bits) if v]
                present_set = set(present)
                overlaps = []
                for i in present:
                    for j in conflict_sets_local[i]:
                        if j in present_set and i < j:
                            overlaps.append((i, j))
                if not overlaps:
                    break
                counts = {}
                for a, b in overlaps:
                    counts[a] = counts.get(a, 0) + 1
                    counts[b] = counts.get(b, 0) + 1
                # remove highest-conflict, then longer edge
                best = None
                best_key = None
                for idx, cnt in counts.items():
                    key = (cnt, lengths[idx])
                    if best_key is None or key > best_key:
                        best_key = key
                        best = idx
                if best is None:
                    break
                bits[best] = 0
                steps += 1
            # final feasibility trimming
            if truss_problem.evaluate_constraints(bits) != 0:
                ones2 = [i for i, v in enumerate(bits) if v]
                ones2.sort(key=lambda x: lengths[x], reverse=True)
                for idx_rm in ones2:
                    bits[idx_rm] = 0
                    if truss_problem.evaluate_constraints(bits) == 0:
                        break
            if truss_problem.evaluate_constraints(bits) == 0:
                return bits
        return None

    # Diversity grid and evaluation cache
    EPS_GRID = 0.02
    # Increase per-cell capacity to preserve more frontier diversity (helps HV)
    MAX_PER_CELL = 5
    def obj_to_cell(obj):
        return (int(obj[0] / EPS_GRID), int(obj[1] / EPS_GRID))

    # Lightweight surrogate prefilter parameters (cheap proxy to avoid many expensive evaluations)
    # SURROGATE_MIN: minimum surrogate score considered promising
    # SURROGATE_SKIP_PROB: probability to skip an evaluation if below SURROGATE_MIN
    SURROGATE_MIN = 0.18
    SURROGATE_SKIP_PROB = 0.5

    eval_cache: Dict[Tuple[int, ...], Tuple[float, float]] = {}
    edge_score = [0.0] * n_bits

    # Quick helpers
    def bitlist_to_str(bits):
        return ''.join('1' if int(x) else '0' for x in bits)

    def conflict_density(bits):
        ones = [i for i, v in enumerate(bits) if v]
        m = len(ones)
        if m < 2:
            return 0.0
        ones_set = set(ones)
        conflicts = 0
        for i in ones:
            for j in conflict_sets[i]:
                if j in ones_set and j > i:
                    conflicts += 1
        denom = m * (m - 1) / 2.0
        return conflicts / denom

    def repair_planarity(bitlist, max_iter=None):
        bl = bitlist[:]
        for idx in fixed_fixed_bits:
            if bl[idx]:
                bl[idx] = 0
        if max_iter is None:
            max_iter = 3 * n_bits
        for _ in range(max_iter):
            ones = [i for i, v in enumerate(bl) if v]
            ones_set = set(ones)
            conflict_pairs = []
            for i in ones:
                for j in conflict_sets[i]:
                    if j in ones_set and j > i:
                        conflict_pairs.append((i, j))
            if not conflict_pairs:
                break
            counts = {}
            for i, j in conflict_pairs:
                counts[i] = counts.get(i, 0) + 1
                counts[j] = counts.get(j, 0) + 1
            remove_idx = max(counts, key=lambda e: (counts[e], edge_len[e]))
            bl[remove_idx] = 0
        return bl

    def densify_planar(bits, target_additions):
        bl = bits[:]
        present = set(i for i, v in enumerate(bl) if v)
        order = sorted(range(n_bits), key=lambda i: -(base_weight[i] + 0.6 * edge_score[i]))
        added = 0
        for idx in order:
            if bl[idx] == 1 or idx in fixed_fixed_bits:
                continue
            conflict = False
            for j in conflict_sets[idx]:
                if bl[j] == 1:
                    conflict = True; break
            if conflict:
                continue
            bl[idx] = 1
            present.add(idx)
            added += 1
            if added >= target_additions:
                break
        return bl

    def greedy_planar_seed(target_edges):
        bl = [0] * n_bits
        present = set()
        order = sorted(range(n_bits), key=lambda i: -(base_weight[i] + 0.4 * edge_score[i]))
        for idx in order:
            if idx in fixed_fixed_bits:
                continue
            ok = True
            for ex in present:
                if idx in conflict_sets[ex]:
                    ok = False; break
            if ok:
                bl[idx] = 1
                present.add(idx)
                if len(present) >= target_edges:
                    break
        return bl

    def star_seed(center_idx, target_degree=4):
        bl = [0] * n_bits
        incident = [k for k, (i, j) in enumerate(bit_to_pair) if i == center_idx or j == center_idx]
        order = sorted(incident, key=lambda k: -(base_weight[k] + 0.5 * edge_score[k]))
        present = set()
        for idx in order:
            if idx in fixed_fixed_bits:
                continue
            ok = True
            for ex in present:
                if idx in conflict_sets[ex]:
                    ok = False; break
            if ok:
                bl[idx] = 1
                present.add(idx)
                if len(present) >= target_degree:
                    break
        return bl

    # Geometry-informed seeds
    def _convex_hull(coords):
        pts = sorted(set(coords))
        if len(pts) <= 2:
            return pts
        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        return lower[:-1] + upper[:-1]

    def seed_convex_hull():
        if len(nodes) < 3:
            return [0] * n_bits
        hull = _convex_hull(nodes)
        if len(hull) < 3:
            return [0] * n_bits
        idx_by_coord = {nodes[i]: i for i in range(len(nodes))}
        bl = [0] * n_bits
        for k in range(len(hull)):
            a = idx_by_coord[hull[k]]
            b = idx_by_coord[hull[(k + 1) % len(hull)]]
            i, j = (a, b) if a < b else (b, a)
            bi = pair_to_bit.get((i, j))
            if bi is not None and bi not in fixed_fixed_bits:
                if not any(bl[c] == 1 for c in conflict_sets[bi]):
                    bl[bi] = 1
        return repair_planarity(bl)

    def seed_boundary_ring():
        # Build an outer boundary ring by selecting nodes near the bounding box perimeter
        bl = [0] * n_bits
        if len(nodes) < 3:
            return bl
        xs = [p[0] for p in nodes]; ys = [p[1] for p in nodes]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        dx = (maxx - minx) + 1e-9
        dy = (maxy - miny) + 1e-9
        boundary = []
        for i in range(len(nodes)):
            x, y = nodes[i]
            if abs(x - minx) <= 0.06*dx or abs(x - maxx) <= 0.06*dx or abs(y - miny) <= 0.06*dy or abs(y - maxy) <= 0.06*dy:
                boundary.append(i)
        boundary = sorted(set(boundary))
        if len(boundary) >= 3:
            cx = sum(nodes[i][0] for i in boundary) / len(boundary)
            cy = sum(nodes[i][1] for i in boundary) / len(boundary)
            boundary.sort(key=lambda i: math.atan2(nodes[i][1]-cy, nodes[i][0]-cx))
            for k in range(len(boundary)):
                a = boundary[k]; b = boundary[(k + 1) % len(boundary)]
                u, v = (a, b) if a < b else (b, a)
                bi = pair_to_bit.get((u, v))
                if bi is None or bi in fixed_fixed_bits or bl[bi] == 1:
                    continue
                if any(bl[c] == 1 for c in conflict_sets[bi]):
                    continue
                bl[bi] = 1
        # add limited braces to nearby anchors (loads/fixed) to improve stiffness potential
        anchors = list(set(load_node_idx + fixed_node_idx))
        if anchors and boundary:
            for a in boundary[:min(6, len(boundary))]:
                nearest = sorted(anchors, key=lambda t: (nodes[a][0]-nodes[t][0])**2 + (nodes[a][1]-nodes[t][1])**2)
                cnt = 0
                for t in nearest[:2]:
                    u, v = (a, t) if a < t else (t, a)
                    bi = pair_to_bit.get((u, v))
                    if bi is None or bi in fixed_fixed_bits or bl[bi] == 1:
                        continue
                    if any(bl[c] == 1 for c in conflict_sets[bi]):
                        continue
                    bl[bi] = 1
                    cnt += 1
                    if cnt >= 2:
                        break
        return repair_planarity(bl)

    def bridge_seed():
        # Connect each load node to its nearest fixed node via a non-conflicting edge
        bl = [0] * n_bits
        if not load_node_idx or not fixed_node_idx:
            return bl
        for ln in load_node_idx:
            nearest = sorted(fixed_node_idx, key=lambda fn: (nodes[ln][0]-nodes[fn][0])**2 + (nodes[ln][1]-nodes[fn][1])**2)
            for fn in nearest:
                a, b = (ln, fn) if ln < fn else (fn, ln)
                idx = pair_to_bit.get((a, b))
                if idx is None or idx in fixed_fixed_bits:
                    continue
                if any(bl[x] == 1 for x in conflict_sets[idx]):
                    continue
                bl[idx] = 1
                break
        return repair_planarity(bl)

    # Grid-aware Pareto insert with per-cell cap (time-adaptive grid resolution and cap)
    def pareto_insert_grid(design_bits, fvals, archive_objs, archive_bits):
        # dominated by any existing?
        for fo in archive_objs:
            if dominates(fo, fvals):
                return False
        # prune dominated by newcomer
        keep_objs = []
        keep_bits = []
        for fo, db in zip(archive_objs, archive_bits):
            if not dominates(fvals, fo):
                keep_objs.append(fo)
                keep_bits.append(db)
        keep_objs.append(fvals)
        keep_bits.append(design_bits)

        archive_objs[:] = keep_objs
        archive_bits[:] = keep_bits

        # adapt grid over time: coarse early (diversity), finer late (intensification)
        try:
            elapsed = time.time() - t0_search
            frac = elapsed / max(ALGORITHM_TIMEOUT_SECONDS, 1.0)
        except Exception:
            frac = 0.0
        eps = 0.03 if frac < 0.35 else (0.025 if frac < 0.65 else 0.02)
        cap = 5 if frac < 0.35 else (4 if frac < 0.65 else 3)

        def cell_of(obj):
            return (int(obj[0] / eps), int(obj[1] / eps))

        cell = cell_of(fvals)
        idxs = [k for k, fo in enumerate(archive_objs) if cell_of(fo) == cell]
        if len(idxs) > cap:
            worst_idx = max(idxs, key=lambda k: (archive_objs[k][0] + archive_objs[k][1]))
            del archive_objs[worst_idx]
            del archive_bits[worst_idx]
        return True

    # Unified evaluation, caching, archive insert, and edge-score update
    def evaluate_and_insert(bits, archive_objs, archive_bits, seen):
        # normalize length and quick geometric repair
        if len(bits) != n_bits:
            bits = (bits + [0] * n_bits)[:n_bits]
        bl = bits[:]

        # quick repair if overlaps
        if conflict_density(bl) > 0.0:
            bl = repair_planarity(bl)

        # optional connectivity enforcement to boost stiffness and feasibility
        if load_node_idx and fixed_node_idx:
            bl = ensure_connectivity(bl, max_add_edges=3)
        if conflict_density(bl) > 0.0:
            bl = repair_planarity(bl)

        # final constraint trimming if needed (greedy remove longest)
        if truss_problem.evaluate_constraints(bl) != 0:
            r = bl[:]
            ones = [i for i, v in enumerate(r) if v]
            ones.sort(key=lambda x: edge_len[x], reverse=True)
            for idx_rm in ones:
                r[idx_rm] = 0
                if truss_problem.evaluate_constraints(r) == 0:
                    break
            bl = r

        if truss_problem.evaluate_constraints(bl) != 0:
            return False

        # Adaptive surrogate prefilter: stricter as archive grows and time advances
        archive_size = len(archive_bits)
        elapsed = time.time() - t0_search
        time_frac = elapsed / max(1.0, ALGORITHM_TIMEOUT_SECONDS)
        surr_min = SURROGATE_MIN + 0.005 * min(archive_size, 20) + 0.08 * max(0.0, time_frac - 0.7)
        surr_skip = min(0.85, SURROGATE_SKIP_PROB + 0.01 * max(0, archive_size - 6))

        surrogate_score = 0.0
        for i, v in enumerate(bl):
            if v:
                surrogate_score += (base_weight[i] + 0.6 * edge_score[i])
        # probabilistically skip low-potential candidates to save expensive evaluations
        if surrogate_score < surr_min and random.random() < surr_skip:
            return False

        # caching and evaluation
        key = tuple(int(x) for x in bl)
        if key in eval_cache:
            neg_s, vol = eval_cache[key]
        else:
            neg_s, vol = truss_problem.evaluate(bl)
            eval_cache[key] = (neg_s, vol)

        bstr = bitlist_to_str(bl)
        if bstr in seen:
            return False

        # insert into Pareto archive with grid capping
        if pareto_insert_grid(bl, (neg_s, vol), archive_objs, archive_bits):
            seen.add(bstr)
            # reward edges present to bias future operations
            for i, v in enumerate(bl):
                if v:
                    edge_score[i] += 1.0
            # mild periodic decay to avoid over-commitment to early edges
            if len(archive_bits) and (len(archive_bits) % 9 == 0):
                for i in range(n_bits):
                    edge_score[i] *= 0.95
            # update last improvement timestamp (mutable list element)
            try:
                last_improve[0] = time.time()
            except Exception:
                pass
            return True
        return False

    # Get static and free node indices
    static_nodes = fixed_node_idx + load_node_idx
    free_nodes = [x for x in all_node_idx if x not in static_nodes]

    # Get all combinations of fixed nodes that include at least two fixed points
    fixed_nodes_pwr_set = list(power_set(fixed_node_idx))
    static_nodes_comb_feasible = []
    for fixed_nodes_comb in fixed_nodes_pwr_set:
        if len(fixed_nodes_comb) > 1:
            static_nodes_comb_feasible.append(list(fixed_nodes_comb) + load_node_idx)

    # Record start time to enforce runtime budget
    t0_search = time.time()

    # Get combinations of free nodes (sample if too many), and shuffle for diversity
    all_free_subsets = list(power_set(free_nodes))
    MAX_FREE_SAMPLES = 1500
    if len(all_free_subsets) > MAX_FREE_SAMPLES:
        free_nodes_pwr_set = random.sample(all_free_subsets, MAX_FREE_SAMPLES)
    else:
        free_nodes_pwr_set = all_free_subsets
    random.shuffle(free_nodes_pwr_set)
    random.shuffle(static_nodes_comb_feasible)

    # Pareto archive and bookkeeping
    archive_bits = []   # list of feasible bit_lists
    archive_objs = []   # list of tuples (neg_stiffness, volume_fraction)
    seen = set()        # bit_str to avoid duplicates
    last_save = t0_search
    # track last improvement time for stagnation detection and reseeding
    # stored as a single-element list so nested functions can update without nonlocal
    last_improve = [t0_search]
    STAGNATION_SEC = 6.5

    # Lightweight reseed when stagnation detected (inject geometry-informed seeds)
    def reseed_if_stagnant():
        now = time.time()
        if now - last_improve[0] < STAGNATION_SEC:
            return
        seeds = []
        try:
            seeds.append(seed_convex_hull())
        except Exception:
            pass
        try:
            seeds.append(seed_boundary_ring())
        except Exception:
            pass
        try:
            seeds.append(bridge_seed())
        except Exception:
            pass
        # small greedy seeds
        for ratio in (0.04, 0.08):
            t_edges = max(1, int(n_bits * ratio))
            seeds.append(greedy_planar_seed(t_edges))
        # Evaluate seeds and light neighborhoods
        for bl in seeds:
            evaluate_and_insert(bl, archive_objs, archive_bits, seen)
            # densify seed slightly
            child = densify_planar(bl, target_additions=1)
            evaluate_and_insert(child, archive_objs, archive_bits, seen)
            # micro-mutation
            mut = mutate_repair(bl, n_bits, truss_problem, k_flips=max(1, int(0.008 * n_bits)))
            if mut is not None:
                evaluate_and_insert(mut, archive_objs, archive_bits, seen)
        last_improve[0] = now

    # Early seeding: greedy planar, hull ring, boundary ring, and star seeds
    for ratio in (0.06, 0.12):
        t_edges = max(1, int(n_bits * ratio))
        evaluate_and_insert(greedy_planar_seed(t_edges), archive_objs, archive_bits, seen)
    try:
        hull_bl = seed_convex_hull()
        evaluate_and_insert(hull_bl, archive_objs, archive_bits, seen)
        evaluate_and_insert(densify_planar(hull_bl, target_additions=2), archive_objs, archive_bits, seen)
    except Exception:
        pass
    try:
        ring_bl = seed_boundary_ring()
        evaluate_and_insert(ring_bl, archive_objs, archive_bits, seen)
    except Exception:
        pass
    for ln in load_node_idx:
        for deg in (3, 5):
            evaluate_and_insert(star_seed(ln, target_degree=deg), archive_objs, archive_bits, seen)
    # Save initial archive if any
    if archive_bits:
        try:
            truss_problem.save_solutions(np.array(archive_bits, dtype=int))
        except Exception:
            pass

    # Iterate over sampled combinations and generate NodeSort seeds + mutations
    for static_nodes_comb in static_nodes_comb_feasible:
        # Timeout check
        if time.time() - t0_search > ALGORITHM_TIMEOUT_SECONDS - 1.0:
            break
        # Stagnation-triggered reseed
        reseed_if_stagnant()
        for free_node_comb in free_nodes_pwr_set:
            if time.time() - t0_search > ALGORITHM_TIMEOUT_SECONDS - 1.0:
                break

            node_comb = list(set(list(free_node_comb) + static_nodes_comb))
            nodes_new = [nodes[idx] for idx in node_comb]

            # Build NodeSort design and convert to bit representation
            edges_new = node_sort_truss(nodes_new)
            bit_list, bit_str, node_idx_pairs, node_coords = truss_problem.convert(edges_new)

            # Remove members connecting fixed nodes and convert again
            node_idx_pairs_fixed = remove_members_connecting_fixed(node_idx_pairs, fixed_node_idx)
            bit_list, bit_str, node_idx_pairs, node_coords = truss_problem.convert(node_idx_pairs_fixed)

            # Skip duplicates quickly
            if bit_str in seen:
                # Periodic save even if duplicate
                if time.time() - last_save > 1.5:
                    if archive_bits:
                        truss_problem.save_solutions(np.array(archive_bits, dtype=int))
                    last_save = time.time()
                continue

            # Conflict-density pruning for tangled NodeSort seeds
            if conflict_density(bit_list) > 0.45 and random.random() < 0.85:
                if time.time() - last_save > 1.5:
                    if archive_bits:
                        truss_problem.save_solutions(np.array(archive_bits, dtype=int))
                    last_save = time.time()
                continue

            # Quick geometric repair and small planar densifications
            bit_list = repair_planarity(bit_list)
            # Enforce minimal connectivity to improve stiffness potential
            if load_node_idx and fixed_node_idx:
                bit_list = ensure_connectivity(bit_list, max_add_edges=3)
            # re-repair after additions
            if conflict_density(bit_list) > 0.0:
                bit_list = repair_planarity(bit_list)
            for d in (0, 2):
                candidate = bit_list if d == 0 else densify_planar(bit_list, target_additions=d)
                evaluate_and_insert(candidate, archive_objs, archive_bits, seen)

            # Explore local neighborhood via small mutations with repair
            for k_mut in (max(1, int(0.005 * n_bits)), max(1, int(0.02 * n_bits))):
                if time.time() - t0_search > ALGORITHM_TIMEOUT_SECONDS - 1.0:
                    break
                mutated = mutate_repair(bit_list, n_bits, truss_problem, k_flips=k_mut)
                if mutated is None:
                    continue
                evaluate_and_insert(mutated, archive_objs, archive_bits, seen)

            # Periodic save of the current archive
            if time.time() - last_save > 1.5:
                if archive_bits:
                    truss_problem.save_solutions(np.array(archive_bits, dtype=int))
                last_save = time.time()

    # Local refinement: leader-centric intensification + restricted mutations until budget drains
    while time.time() - t0_search < ALGORITHM_TIMEOUT_SECONDS - 1.0:
        # reseed periodically if no improvement
        reseed_if_stagnant()

        # Intensify around leaders (best stiffness, best volume, and most crowded)
        if archive_bits and archive_objs:
            try:
                idx_best_stiff = min(range(len(archive_objs)), key=lambda k: archive_objs[k][0])
                idx_best_vol = min(range(len(archive_objs)), key=lambda k: archive_objs[k][1])
                crowd = compute_crowding_priority(archive_objs)
                idx_best_crowd = max(range(len(crowd)), key=lambda k: crowd[k]) if crowd else idx_best_stiff
                for li in {idx_best_stiff, idx_best_vol, idx_best_crowd}:
                    parent = archive_bits[li]
                    # slight densification near leader
                    child_d = densify_planar(parent, target_additions=1)
                    evaluate_and_insert(child_d, archive_objs, archive_bits, seen)
                    # two-hop restricted neighborhood mutations (focused exploitation)
                    allowed = get_two_hop_allowed_indices(parent, depth=2, max_nodes=40)
                    for _ in range(2):
                        child_loc = mutate_subset_repair(parent, allowed, n_bits, truss_problem, k_flips=max(1, int(0.01 * n_bits)))
                        if child_loc is not None:
                            evaluate_and_insert(child_loc, archive_objs, archive_bits, seen)
                    # small global mutation near leader
                    child_m = mutate_repair(parent, n_bits, truss_problem, k_flips=max(1, int(0.015 * n_bits)))
                    if child_m is not None:
                        evaluate_and_insert(child_m, archive_objs, archive_bits, seen)
            except Exception:
                pass

        # Baseline sweep across current archive
        for bits in list(archive_bits):
            if time.time() - t0_search > ALGORITHM_TIMEOUT_SECONDS - 1.0:
                break
            mutated = mutate_repair(bits, n_bits, truss_problem, k_flips=max(1, int(0.02 * n_bits)))
            if mutated is not None:
                evaluate_and_insert(mutated, archive_objs, archive_bits, seen)
            densified = densify_planar(bits, target_additions=1)
            if densified is not None:
                evaluate_and_insert(densified, archive_objs, archive_bits, seen)
        # continue until time runs out
    # Final save of the Pareto archive
    if archive_bits:
        truss_problem.save_solutions(np.array(archive_bits))
    else:
        # Fallback: save empty to avoid downstream errors
        truss_problem.save_solutions(np.array([]))


def power_set(iterable):
    """
    Returns an iterator over all subsets (the power set) of the iterable.
    The subsets are returned as tuples, including the empty tuple ().
    """
    s = list(iterable)
    # Generates combinations for all possible lengths r from 0 to len(s)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

def node_sort_truss(nodes):
    """
    Implements the NodeSort truss design algorithm with corrected
    boundary processing and triangulation logic. Places triangular structures considering given nodes.

    Args:
        nodes (list of tuple): A list of nodes, where each node is an (x, y) tuple.

    Returns:
        list of tuple: A list of edges representing the truss design.
                       ((x1, y1), (x2, y2))
    """
    # 1. Sort nodes ascending over x, if equal compare over y
    sorted_nodes = sorted(nodes)
    n = len(sorted_nodes)

    edges = []

    # 2. Point to first node i = 0
    # 3. Loop while i < (n_i - 1)
    # Correction: The loop must process the second-to-last node
    # to ensure the final segment is connected.
    # range(n-1) goes from 0 to n-2.
    for i in range(n - 1):
        node_i = sorted_nodes[i]

        # We start checking from the very next node
        next_node_index = i + 1
        if next_node_index >= n:
            break

        node_next = sorted_nodes[next_node_index]

        targets = []

        # Determine if we are in Case A (Next node is equal/lower) or B (Next node is higher)
        # Note: The prompt says "node_i.y >= node_{i+1}.y" for A
        if node_i[1] >= node_next[1]:
            # --- CASE A: Ascending Search ---
            above_count = 0

            for j in range(i + 1, n):
                candidate = sorted_nodes[j]

                # Check Break Criterion (a): Ordering
                # We compare current candidate with the PREVIOUSLY collected target
                # to ensure the sequence of targets is ascending over y.
                if targets:
                    prev_target = targets[-1]
                    # If candidate drops below the previous target, ordering is broken
                    if candidate[1] < prev_target[1]:
                        break

                        # Check Break Criterion (b): Count of nodes above node_i
                # We collect the node first, then check if we should stop future searches.
                is_above = candidate[1] > node_i[1]

                if is_above:
                    above_count += 1

                # Add the candidate to the truss
                targets.append(candidate)

                # If this was the second node above, we break *after* adding it.
                # This ensures we capture the closing edge of the triangle.
                if above_count == 2:
                    break

        else:
            # --- CASE B: Descending Search ---
            below_count = 0

            for j in range(i + 1, n):
                candidate = sorted_nodes[j]

                # Check Break Criterion (a): Ordering (Descending)
                if targets:
                    prev_target = targets[-1]
                    # If candidate rises above the previous target, ordering is broken
                    if candidate[1] > prev_target[1]:
                        break

                # Check Break Criterion (b): Count of nodes below node_i
                is_below = candidate[1] < node_i[1]

                if is_below:
                    below_count += 1

                targets.append(candidate)

                if below_count == 2:
                    break

        # C. Define trusses
        for target in targets:
            edges.append((node_i, target))

    return edges

def remove_members_connecting_fixed(node_idx_pairs, fixed_nodes_idx):
    node_idx_pairs_new = []
    for node_idx_pair in node_idx_pairs:
        node1, node2 = node_idx_pair
        if node1 in fixed_nodes_idx and node2 in fixed_nodes_idx:
            continue
        else:
            node_idx_pairs_new.append(node_idx_pair)
    return node_idx_pairs_new





# EVOLVE-BLOCK-END






if __name__ == '__main__':

    # from combench.models.truss import train_problems, val_problems
    # v_problem = val_problems[2]
    val_num = 1
    # for val_num in range(2, 8):
    # from combench.models.truss.problems.cantilever import get_problems
    # train_problems, val_problems, val_problems_out = get_problems()
    # v_problem = val_problems[val_num]
    # truss.set_norms(v_problem)
    # from combench.nn.trussDecoderUMD import problem as v_problem
    from tests.models.test_cantilever import g_problem

    # Truss Problem
    truss.set_norms(g_problem)
    p_model = TrussModel(g_problem)


    # Algorithm
    print('running shinka algorithm...')
    search_algorithm(p_model)
    print('done')
    designs = p_model.load_solutions()
    designs = designs.tolist()
    print('Designs:', designs)


    pop_size = 200
    ref_point = np.array([0, 1])
    pop = TrussPopulation(pop_size, ref_point, p_model)
    max_nfe = 100000

    for design in designs:
        truss_design = TrussDesign(design, p_model)
        pop.add_design(truss_design)
        # print('New design:', truss_design)

    pop.prune()
    hypervolume = pop.calc_hv()
    print('Hypervolume:', hypervolume)

    save_dir = '/Users/gapaza/repos/ideal/combench/plots/shinka/alg4'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    pop.plot_population(save_dir)









