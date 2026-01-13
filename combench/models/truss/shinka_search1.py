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



# -----------------------------------------------------------
# SHINKA ALGORITHM
# -----------------------------------------------------------
# EVOLVE-BLOCK-START
import random
from itertools import chain, combinations
import math

# Keep the global timeout constant
ALGORITHM_TIMEOUT_SECONDS = 240  # seconds

def search_algorithm(truss_problem):
    """
    Adaptive HV-guided frontier-focused search for truss Pareto front.
    Saves nondominated feasible designs via truss_problem.save_solutions(np.ndarray).
    """
    # ---------- Helpers ----------
    def dominates(a, b):
        return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])

    # geometry helpers: orientation / segment intersection
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

    # ---------- Problem Setup ----------
    pf = truss_problem.get_problem_formulation()
    nodes = pf['nodes']
    all_node_idx = list(range(len(nodes)))
    fixed_node_idx = truss_problem.get_fixed_nodes()
    load_node_idx = truss_problem.get_load_nodes()
    n_bits = truss_problem.get_n_bits()

    # map bits <-> node pairs
    bit_to_pair = []
    pair_to_bit = {}
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            idx = len(bit_to_pair)
            bit_to_pair.append((i,j))
            pair_to_bit[(i,j)] = idx
    coords_by_bit = [(nodes[i], nodes[j]) for (i,j) in bit_to_pair]
    edge_len = [math.hypot(a[0]-b[0], a[1]-b[1]) for (a,b) in coords_by_bit]

    # produce conflict sets once
    conflict_sets = [set() for _ in range(n_bits)]
    for a in range(n_bits):
        a0, a1 = coords_by_bit[a]
        for b in range(a+1, n_bits):
            b0, b1 = coords_by_bit[b]
            if segments_intersect(a0, a1, b0, b1):
                conflict_sets[a].add(b)
                conflict_sets[b].add(a)

    fixed_fixed_bits = set(idx for idx, (i,j) in enumerate(bit_to_pair) if i in fixed_node_idx and j in fixed_node_idx)

    # base weight favors short edges and incident to load/fixed nodes
    base_weight = []
    fixed_set_tmp = set(fixed_node_idx)
    load_set_tmp = set(load_node_idx)
    for idx2, (ii, jj) in enumerate(bit_to_pair):
        w = 1.0 / (edge_len[idx2] + 1e-9)
        if ii in load_set_tmp or jj in load_set_tmp:
            w += 2.5
        if ii in fixed_set_tmp or jj in fixed_set_tmp:
            w += 1.2
        base_weight.append(w)


    # ---------- State and parameters ----------
    eval_cache = {}  # map tuple(bits)->(neg_stiff, vol)
    edge_score = [0.0]*n_bits  # learned edge rewards
    archive_bits = []   # list of bit-lists for nondominated
    archive_objs = []   # corresponding objective tuples
    seen = set()        # bitstring dedupe
    reservoir = {}      # per-cell reservoir for dominated feasibles
    reservoir_seen = set()

    # diversity grid / reservoir params
    EPS_GRID = 0.02
    RESERVOIR_PER_CELL = 3
    def obj_to_cell(obj): return (int(obj[0]/EPS_GRID), int(obj[1]/EPS_GRID))

    # adaptive pruning / surrogate params
    INITIAL_PRUNE = 0.45
    PRUNE_RELAX = 0.60
    PRUNE_TIGHTEN = 0.40
    prune_thr = INITIAL_PRUNE
    SURROGATE_BASE_MIN = 0.18
    SURROGATE_SKIP_PROB = 0.45

    # stagnation detection
    t0_search = time.time()
    last_improve = [t0_search]
    STAG_SEC = 6.0

    last_save = t0_search

    # HV tracking
    BEST_HV = [0.0]
    LAST_HV_IMPROVE = [t0_search]

    # sampling limits
    def power_set(iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

    free_nodes = [x for x in all_node_idx if x not in (fixed_node_idx + load_node_idx)]
    all_free_subsets = list(power_set(free_nodes))
    MAX_FREE_SAMPLES = 1500
    if len(all_free_subsets) > MAX_FREE_SAMPLES:
        free_nodes_pwr_set = random.sample(all_free_subsets, MAX_FREE_SAMPLES)
    else:
        free_nodes_pwr_set = all_free_subsets
    random.shuffle(free_nodes_pwr_set)

    fixed_nodes_pwr_set = list(power_set(fixed_node_idx))
    static_nodes_comb_feasible = []
    for comb in fixed_nodes_pwr_set:
        if len(comb) > 1:
            static_nodes_comb_feasible.append(list(comb) + load_node_idx)
    random.shuffle(static_nodes_comb_feasible)

    # ---------- Utility functions ----------
    def bitlist_to_str(bits):
        return ''.join('1' if int(x) else '0' for x in bits)

    def conflict_density(bitlist):
        ones = [i for i,v in enumerate(bitlist) if v]
        m = len(ones)
        if m < 2:
            return 0.0
        ones_set = set(ones)
        conflicts = 0
        for i in ones:
            for j in conflict_sets[i]:
                if j in ones_set and j > i:
                    conflicts += 1
        denom = m*(m-1)/2.0
        return conflicts/denom if denom>0 else 0.0

    def repair_planarity(bitlist, max_iter=None):
        bl = bitlist[:]
        for idx in fixed_fixed_bits:
            if bl[idx]:
                bl[idx] = 0
        if max_iter is None:
            max_iter = 3 * n_bits
        for _ in range(max_iter):
            ones = [i for i,v in enumerate(bl) if v]
            ones_set = set(ones)
            conflict_pairs = []
            for i in ones:
                for j in conflict_sets[i]:
                    if j in ones_set and j > i:
                        conflict_pairs.append((i,j))
            if not conflict_pairs:
                break
            counts = {}
            for a,b in conflict_pairs:
                counts[a] = counts.get(a,0)+1
                counts[b] = counts.get(b,0)+1
            remove_idx = max(counts, key=lambda e: (counts[e], edge_len[e]))
            bl[remove_idx] = 0
        return bl

    def densify_planar(bits, target_additions=2):
        bl = bits[:]
        order = sorted(range(n_bits), key=lambda i: -(base_weight[i] + 0.6*edge_score[i]))
        added = 0
        for idx in order:
            if bl[idx] or idx in fixed_fixed_bits:
                continue
            conflict = any(bl[j] for j in conflict_sets[idx])
            if conflict:
                continue
            bl[idx] = 1
            added += 1
            if added >= target_additions: break
        return bl

    def greedy_planar_seed(target_edges):
        bl = [0]*n_bits
        present = set()
        order = sorted(range(n_bits), key=lambda i: -(base_weight[i] + 0.5*edge_score[i] + random.random()*1e-3))
        for idx in order:
            if idx in fixed_fixed_bits: continue
            ok = True
            for ex in present:
                if idx in conflict_sets[ex]:
                    ok = False; break
            if ok:
                bl[idx] = 1
                present.add(idx)
                if len(present) >= target_edges: break
        return bl

    def star_seed(center_idx, target_degree=4):
        bl = [0]*n_bits
        incident = [k for k,(i,j) in enumerate(bit_to_pair) if i==center_idx or j==center_idx]
        order = sorted(incident, key=lambda k: -(base_weight[k] + 0.5*edge_score[k]))
        present = set()
        for idx in order:
            if idx in fixed_fixed_bits: continue
            ok = True
            for ex in present:
                if idx in conflict_sets[ex]:
                    ok = False; break
            if ok:
                bl[idx] = 1
                present.add(idx)
                if len(present) >= target_degree: break
        return bl

    # geometry-informed seeds: convex hull, boundary ring, bridge
    def seed_convex_hull():
        bl = [0]*n_bits
        if not nodes: return bl
        pts = [(nodes[i][0], nodes[i][1], i) for i in range(len(nodes))]
        pts_sorted = sorted(pts)
        def cross(o,a,b): return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        lower = []
        for p in pts_sorted:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in reversed(pts_sorted):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        hull = lower[:-1] + upper[:-1]
        if not hull: return bl
        hull_idx = [c for _,_,c in hull]
        for i in range(len(hull_idx)):
            a = hull_idx[i]; b = hull_idx[(i+1)%len(hull_idx)]
            u,v = (a,b) if a<b else (b,a)
            idx = pair_to_bit.get((u,v))
            if idx is None or idx in fixed_fixed_bits: continue
            if any(bl[x] for x in conflict_sets[idx]): continue
            bl[idx] = 1
        # spokes from loads to hull
        for ln in load_node_idx:
            if not hull_idx or ln in hull_idx: continue
            nearest = sorted(hull_idx, key=lambda h: (nodes[h][0]-nodes[ln][0])**2 + (nodes[h][1]-nodes[ln][1])**2)
            added = 0
            for h in nearest[:2]:
                u,v = (ln,h) if ln<h else (h,ln)
                idx = pair_to_bit.get((u,v))
                if idx is None or idx in fixed_fixed_bits or bl[idx]: continue
                if any(bl[x] for x in conflict_sets[idx]): continue
                bl[idx]=1; added+=1
                if added>=2: break
        return repair_planarity(bl)

    def seed_boundary_ring():
        bl = [0]*n_bits
        n = len(nodes)
        if n<2: return bl
        xs = [nodes[i][0] for i in range(n)]; ys = [nodes[i][1] for i in range(n)]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        dx = maxx-minx + 1e-9; dy = maxy-miny + 1e-9
        boundary = []
        for i in range(n):
            x,y = nodes[i]
            if abs(x-minx)<=0.06*dx or abs(x-maxx)<=0.06*dx or abs(y-miny)<=0.06*dy or abs(y-maxy)<=0.06*dy:
                boundary.append(i)
        if len(boundary)>=2:
            cx = sum(nodes[i][0] for i in boundary)/len(boundary)
            cy = sum(nodes[i][1] for i in boundary)/len(boundary)
            boundary.sort(key=lambda i: math.atan2(nodes[i][1]-cy, nodes[i][0]-cx))
            for k in range(len(boundary)):
                a = boundary[k]; b = boundary[(k+1)%len(boundary)]
                u,v = (a,b) if a<b else (b,a)
                idx = pair_to_bit.get((u,v))
                if idx is None or idx in fixed_fixed_bits: continue
                if any(bl[x] for x in conflict_sets[idx]): continue
                bl[idx] = 1
        # braces toward anchors
        anchors = list(set(load_node_idx + fixed_node_idx))
        if anchors and boundary:
            for a in boundary[:min(len(boundary),6)]:
                nearest = sorted(anchors, key=lambda t: (nodes[a][0]-nodes[t][0])**2 + (nodes[a][1]-nodes[t][1])**2)
                cnt=0
                for t in nearest[:2]:
                    u,v = (a,t) if a<t else (t,a)
                    idx = pair_to_bit.get((u,v))
                    if idx is None or idx in fixed_fixed_bits or bl[idx]: continue
                    if any(bl[x] for x in conflict_sets[idx]): continue
                    bl[idx]=1; cnt+=1
                    if cnt>=2: break
        return repair_planarity(bl)

    def bridge_seed():
        bl = [0]*n_bits
        if not load_node_idx or not fixed_node_idx:
            return bl
        for ln in load_node_idx:
            nearest = sorted(fixed_node_idx, key=lambda fn: (nodes[ln][0]-nodes[fn][0])**2 + (nodes[ln][1]-nodes[fn][1])**2)
            for fn in nearest:
                a, b = (ln, fn) if ln < fn else (fn, ln)
                idx = pair_to_bit.get((a, b))
                if idx is None or idx in fixed_fixed_bits:
                    continue
                if any(bl[x] for x in conflict_sets[idx]):
                    continue
                bl[idx] = 1
                break
        return repair_planarity(bl)

    # reservoir helpers
    def reservoir_add(bits, obj):
        cell = obj_to_cell(obj)
        bstr = bitlist_to_str(bits)
        if bstr in reservoir_seen: return
        lst = reservoir.get(cell, [])
        if len(lst) < RESERVOIR_PER_CELL:
            lst.append(bits[:]); reservoir[cell] = lst; reservoir_seen.add(bstr)
        else:
            worst_idx = None; worst_score = -1.0
            for idx,cand in enumerate(lst):
                k = tuple(int(x) for x in cand)
                nv,vv = eval_cache.get(k, (float('inf'), float('inf')))
                s = nv+vv
                if s > worst_score:
                    worst_score = s; worst_idx = idx
            if worst_idx is not None:
                lst[worst_idx] = bits[:]; reservoir[cell] = lst; reservoir_seen.add(bstr)

    def reservoir_sample():
        flat = []
        for lst in reservoir.values():
            flat.extend(lst)
        if not flat:
            return None
        # Prefer candidates from objective-space cells that are underrepresented in the archive
        arch_counts = {}
        for fo in archive_objs:
            c = obj_to_cell(fo)
            arch_counts[c] = arch_counts.get(c, 0) + 1
        scored = []
        for cand in flat:
            key = tuple(int(x) for x in cand)
            if key not in eval_cache:
                continue
            obj = eval_cache[key]
            c = obj_to_cell(obj)
            scarcity = 1.0 / (arch_counts.get(c, 0) + 1)  # higher if cell is sparse in archive
            # light novelty via L1 objective distance to closest archive point (cached)
            if archive_objs:
                dmin = min(abs(obj[0] - fo[0]) + abs(obj[1] - fo[1]) for fo in archive_objs)
            else:
                dmin = 0.0
            score = 2.0 * scarcity + 0.2 * dmin
            scored.append((score, cand))
        if scored:
            scored.sort(key=lambda t: -t[0])
            return scored[0][1]
        # Fallback if no cached objs available
        return random.choice(flat)

    # connectivity helpers: ensure loads are connected to fixed nodes
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
                    if bl[idx]:
                        continue
                    # conflict check
                    conflict = False
                    for ex in conflict_sets[idx]:
                        if bl[ex]:
                            conflict = True; break
                    if conflict:
                        continue
                    score = base_weight[idx] + 0.2/(edge_len[idx] + 1e-9)
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

        # final safety: planarity repair
        if conflict_density(bl) > 0.0:
            bl = repair_planarity(bl)
        return bl

    # Adaptive prune update based on recent inserts
    recent_inserts = []
    RECENT_WINDOW = 10
    def update_prune(on_insert):
        nonlocal prune_thr
        recent_inserts.append(1 if on_insert else 0)
        if len(recent_inserts) > RECENT_WINDOW:
            recent_inserts.pop(0)
        avg = sum(recent_inserts)/len(recent_inserts)
        if avg < 0.15:
            prune_thr = PRUNE_RELAX
        elif avg > 0.35:
            prune_thr = PRUNE_TIGHTEN
        else:
            prune_thr = INITIAL_PRUNE

    # ---------- Global conflict-aware mutate + repair (from inspiration) ----------
    def mutate_repair(base_bits, n_bits_local, truss_problem_local, k_flips=0, max_repair_steps=40, max_tries=2):
        if k_flips <= 0:
            k_flips = max(1, int(0.01 * n_bits_local))
        # cache geometry/conflicts per problem
        state = getattr(mutate_repair, "_state", None)
        if state is None:
            pf_local = truss_problem_local.get_problem_formulation()
            nodes_local = pf_local['nodes']
            bit_to_pair_local = []
            coords_by_bit_local = []
            for i in range(len(nodes_local)):
                for j in range(i + 1, len(nodes_local)):
                    bit_to_pair_local.append((i, j))
                    coords_by_bit_local.append((nodes_local[i], nodes_local[j]))
            m = len(bit_to_pair_local)
            lengths = [math.hypot(a[0]-b[0], a[1]-b[1]) for (a, b) in coords_by_bit_local]
            def orient_l(a, b, c):
                return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
            def on_segment_l(a, b, p):
                return (min(a[0], b[0]) - 1e-9 <= p[0] <= max(a[0], b[0]) + 1e-9 and
                        min(a[1], b[1]) - 1e-9 <= p[1] <= max(a[1], b[1]) + 1e-9)
            def segments_intersect_l(a, b, c, d):
                if a == c or a == d or b == c or b == d:
                    return False
                if max(a[0], b[0]) + 1e-9 < min(c[0], d[0]) - 1e-9 or max(c[0], d[0]) + 1e-9 < min(a[0], b[0]) - 1e-9:
                    return False
                if max(a[1], b[1]) + 1e-9 < min(c[1], d[1]) - 1e-9 or max(c[1], d[1]) + 1e-9 < min(a[1], b[1]) - 1e-9:
                    return False
                o1 = orient_l(a,b,c); o2 = orient_l(a,b,d); o3 = orient_l(c,d,a); o4 = orient_l(c,d,b)
                if abs(o1) < 1e-12 and on_segment_l(a,b,c): return True
                if abs(o2) < 1e-12 and on_segment_l(a,b,d): return True
                if abs(o3) < 1e-12 and on_segment_l(c,d,a): return True
                if abs(o4) < 1e-12 and on_segment_l(c,d,b): return True
                if (o1>0 and o2<0 or o1<0 and o2>0) and (o3>0 and o4<0 or o3<0 and o4>0):
                    return True
                return False
            conflict_sets_local = [set() for _ in range(m)]
            for i in range(m):
                a, b = coords_by_bit_local[i]
                for j in range(i+1, m):
                    c, d = coords_by_bit_local[j]
                    if segments_intersect_l(a, b, c, d):
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
            # prefer add/remove by learned+base weight where possible
            zeros = [i for i, v in enumerate(bits) if v == 0]
            ones = [i for i, v in enumerate(bits) if v == 1]
            add_k = min(len(zeros), max(1, int(0.5 * k_flips)))
            rem_k = min(len(ones), max(0, k_flips - add_k))
            chosen = set()
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
                    chosen.add(idx); bits[idx] = 1
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
                    chosen.add(idx); bits[idx] = 0
            # fill remaining flips randomly
            current_flips = len(chosen)
            if current_flips < min(k_flips, n_bits_local):
                remaining = [i for i in range(n_bits_local) if i not in chosen]
                need = min(min(k_flips, n_bits_local) - current_flips, len(remaining))
                if need > 0:
                    for idx in random.sample(remaining, need):
                        bits[idx] = 1 - bits[idx]

            # repair overlaps
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

            # final feas check: greedy remove longest until feasible
            if truss_problem_local.evaluate_constraints(bits) != 0:
                ones2 = [i for i, v in enumerate(bits) if v]
                ones2.sort(key=lambda x: lengths[x], reverse=True)
                for idx_rm in ones2:
                    bits[idx_rm] = 0
                    if truss_problem_local.evaluate_constraints(bits) == 0:
                        break
            if truss_problem_local.evaluate_constraints(bits) == 0:
                return bits
        return None

    # ---------- Local search operators (frontier-centric) ----------
    # Localized mutation: prefer flips near present edges (within 2 hops)
    def localized_mutation(bits, k_flips):
        bl = bits[:]
        present = [i for i,v in enumerate(bl) if v]
        if not present:
            # fallback random flips
            idxs = random.sample(range(n_bits), min(k_flips, n_bits))
            for idx in idxs: bl[idx] = 1-bl[idx]
            bl = repair_planarity(bl)
            try:
                bl = ensure_connectivity(bl, max_add_edges=2)
            except Exception:
                pass
            if conflict_density(bl) > 0.0:
                bl = repair_planarity(bl)
            return bl
        # gather neighborhood: incident edges
        neigh = set(present)
        for e in present:
            a,b = bit_to_pair[e]
            for idx,(i,j) in enumerate(bit_to_pair):
                if i==a or j==a or i==b or j==b:
                    neigh.add(idx)
        # select flips biased: add high-scoring zeros, remove low-scoring ones
        zeros = [i for i in neigh if bl[i]==0 and i not in fixed_fixed_bits]
        ones = [i for i in neigh if bl[i]==1]
        flips = []
        add_k = max(1, int(0.6*k_flips))
        rem_k = max(0, k_flips - add_k)
        # additions weighted by score
        if zeros:
            weights = [edge_score[i] + 0.4*base_weight[i] + 1e-8 for i in zeros]
            s = sum(weights)
            if s>0:
                probs = [w/s for w in weights]
                chosen = np_random_choice_without_replace(zeros, min(len(zeros), add_k), probs)
            else:
                chosen = random.sample(zeros, min(len(zeros), add_k))
            for c in chosen:
                flips.append(c); bl[c]=1
        if ones and rem_k>0:
            weights = [edge_score[i] + 0.4*base_weight[i] + 1e-8 for i in ones]
            inv = [1.0/w for w in weights]
            s2 = sum(inv)
            if s2>0:
                probs2 = [w/s2 for w in inv]
                chosen_rem = np_random_choice_without_replace(ones, min(len(ones), rem_k), probs2)
            else:
                chosen_rem = random.sample(ones, min(len(ones), rem_k))
            for c in chosen_rem:
                flips.append(c); bl[c]=0
        # if flips insufficient, random elsewhere
        if len(flips) < k_flips:
            remaining = [i for i in range(n_bits) if i not in flips and i not in fixed_fixed_bits]
            need = min(k_flips - len(flips), len(remaining))
            for idx in random.sample(remaining, need):
                bl[idx] = 1-bl[idx]
        bl = repair_planarity(bl)
        # light connectivity enforcement
        try:
            bl = ensure_connectivity(bl, max_add_edges=2)
        except Exception:
            pass
        if conflict_density(bl) > 0.0:
            bl = repair_planarity(bl)
        return bl

    # helper wrapper to sample without replace with probabilities using numpy if available
    def np_random_choice_without_replace(pool, k, probs):
        try:
            import numpy as _np
            return list(_np.random.choice(pool, size=k, replace=False, p=probs))
        except Exception:
            return random.sample(pool, k)

    # Diversity proxy in bit-space: 1 - max Jaccard similarity to a small sample of archive designs
    def novelty_score_bits(bl, sample_k=8):
        if not archive_bits:
            return 1.0
        try:
            sample = random.sample(archive_bits, min(sample_k, len(archive_bits)))
        except Exception:
            sample = archive_bits[:sample_k]
        bl_set = set(i for i, v in enumerate(bl) if v)
        max_jacc = 0.0
        for other in sample:
            other_set = set(i for i, v in enumerate(other) if v)
            inter = len(bl_set & other_set)
            union = len(bl_set | other_set)
            j = (inter / union) if union else 0.0
            if j > max_jacc:
                max_jacc = j
        return 1.0 - max_jacc

    # Leader two-hop neighborhood utilities
    def two_hop_allowed_indices(parent_bits, max_nodes=14, hops=2):
        adj = [[] for _ in range(len(nodes))]
        deg = [0] * len(nodes)
        used = set()
        present_edges = [i for i, v in enumerate(parent_bits) if v]
        for e in present_edges:
            u, v = bit_to_pair[e]
            adj[u].append(v); adj[v].append(u)
            deg[u] += 1; deg[v] += 1
            used.add(u); used.add(v)
        centers = set()
        if used:
            top_nodes = sorted(list(used), key=lambda x: (-deg[x], x))[:2]
            centers.update(top_nodes)
        centers.update(load_node_idx)
        centers.update(fixed_node_idx)
        from collections import deque
        dq = deque()
        visited = set()
        for c in centers:
            dq.append((c, 0)); visited.add(c)
        while dq and len(visited) < max_nodes:
            x, d = dq.popleft()
            if d >= hops:
                continue
            for y in adj[x]:
                if y not in visited:
                    visited.add(y)
                    dq.append((y, d + 1))
                if len(visited) >= max_nodes:
                    break
        if len(visited) < max_nodes and len(nodes) > len(visited):
            remaining = [i for i in range(len(nodes)) if i not in visited]
            def min_dist_to_centers(i):
                xi, yi = nodes[i]
                best = 1e18
                for c in centers:
                    xc, yc = nodes[c]
                    d = (xi - xc) * (xi - xc) + (yi - yc) * (yi - yc)
                    if d < best: best = d
                return best
            remaining.sort(key=min_dist_to_centers)
            for r in remaining:
                visited.add(r)
                if len(visited) >= max_nodes:
                    break
        vset = set(visited)
        allowed = []
        for idx, (u, v) in enumerate(bit_to_pair):
            if u in vset and v in vset:
                allowed.append(idx)
        return set(allowed)

    def local_mutate_repair_subset(base_bits, allowed_idx_set, k_add=2, k_rem=1, max_repair_steps=25, max_tries=1):
        for _ in range(max_tries):
            bl = list(base_bits)
            zeros = [i for i in allowed_idx_set if bl[i]==0 and i not in fixed_fixed_bits]
            ones  = [i for i in allowed_idx_set if bl[i]==1]
            add_cnt = min(len(zeros), max(1, int(k_add)))
            rem_cnt = min(len(ones), max(0, int(k_rem)))
            if add_cnt > 0 and zeros:
                try:
                    w = np.array([edge_score[i] + 0.4*base_weight[i] + 1e-8 for i in zeros], dtype=float)
                    s = w.sum()
                    if s > 0:
                        probs = w / s
                        picks = list(np.random.choice(zeros, size=add_cnt, replace=False, p=probs))
                    else:
                        picks = random.sample(zeros, add_cnt)
                except Exception:
                    picks = random.sample(zeros, add_cnt)
                for i in picks:
                    bl[i] = 1
            if rem_cnt > 0 and ones:
                try:
                    w = np.array([edge_score[i] + 0.4*base_weight[i] + 1e-8 for i in ones], dtype=float)
                    inv = 1.0 / (w + 1e-9)
                    inv = inv / inv.sum()
                    picks_rem = list(np.random.choice(ones, size=rem_cnt, replace=False, p=inv))
                except Exception:
                    picks_rem = random.sample(ones, rem_cnt)
                for i in picks_rem:
                    bl[i] = 0
            # overlap repair
            steps = 0
            while steps < max_repair_steps:
                present = [i for i,v in enumerate(bl) if v]
                pset = set(present)
                overlaps = []
                for i in present:
                    for j in conflict_sets[i]:
                        if j in pset and i < j:
                            overlaps.append((i,j))
                if not overlaps:
                    break
                scores = {}
                for a,b in overlaps:
                    scores[a] = scores.get(a,0) + 1
                    scores[b] = scores.get(b,0) + 1
                cand = None; cand_key = None
                for idx,cnt in scores.items():
                    key = (cnt, edge_len[idx], -edge_score[idx])
                    if cand_key is None or key > cand_key:
                        cand_key = key; cand = idx
                if cand is None:
                    break
                bl[cand] = 0
                steps += 1
            try:
                bl = ensure_connectivity(bl, max_add_edges=2)
            except Exception:
                pass
            if conflict_density(bl) > 0.0:
                bl = repair_planarity(bl)
            if truss_problem.evaluate_constraints(bl) == 0:
                return bl
        return None

    # frontier local intensification (budgeted)
    def frontier_local_refine(leaders, budget_iters=6):
        for leader in leaders:
            for it in range(budget_iters):
                k = max(1, int(0.01*n_bits))
                child = localized_mutation(leader, k)
                evaluate_and_insert(child)
                if random.random() < 0.3:
                    child2 = densify_planar(child, target_additions=1)
                    evaluate_and_insert(child2)
                try:
                    # Expand neighborhood scope adaptively based on HV stagnation
                    stagnation = time.time() - LAST_HV_IMPROVE[0]
                    h = 3 if stagnation > 10.0 else 2
                    allowed = two_hop_allowed_indices(leader, max_nodes=16 if h == 3 else 14, hops=h)
                    for _ in range(2):
                        loc = local_mutate_repair_subset(leader, allowed_idx_set=allowed, k_add=2, k_rem=1, max_repair_steps=20, max_tries=1)
                        if loc is not None:
                            evaluate_and_insert(loc)
                except Exception:
                    pass

    # ---------- Evaluation & insertion ----------
    def pareto_insert_grid(des_bits, fvals, archive_objs_local, archive_bits_local):
        # reject if dominated
        for fo in archive_objs_local:
            if dominates(fo, fvals):
                return False
        # remove any dominated by newcomer
        keep_o = []
        keep_b = []
        for fo,db in zip(archive_objs_local, archive_bits_local):
            if not dominates(fvals, fo):
                keep_o.append(fo); keep_b.append(db)
        keep_o.append(fvals); keep_b.append(des_bits)
        archive_objs_local[:] = keep_o; archive_bits_local[:] = keep_b

        # time-adaptive grid resolution and per-cell cap
        try:
            elapsed = time.time() - t0_search
            frac = max(0.0, min(1.0, elapsed / max(1.0, ALGORITHM_TIMEOUT_SECONDS)))
        except Exception:
            frac = 0.0
        eps = 0.03 if frac < 0.35 else (0.025 if frac < 0.65 else 0.02)
        cap = 5 if frac < 0.35 else (4 if frac < 0.65 else 3)

        def cell_of(obj):
            return (int(obj[0] / eps), int(obj[1] / eps))

        # cap per cell for the newcomer's cell
        cell = cell_of(fvals)
        idxs = [k for k, fo in enumerate(archive_objs_local) if cell_of(fo) == cell]
        if len(idxs) > cap:
            worst_idx = max(idxs, key=lambda k: (archive_objs_local[k][0] + archive_objs_local[k][1]))
            del archive_objs_local[worst_idx]; del archive_bits_local[worst_idx]
        return True

    def hv_of_front(frontier_pts):
        if not frontier_pts:
            return 0.0
        # reference point set to slightly beyond current maxima (minimization objectives)
        ref1 = max(p[0] for p in frontier_pts) + 1.0
        ref2 = max(p[1] for p in frontier_pts) + 1.0
        pts = sorted(frontier_pts, key=lambda t: t[0])
        hv = 0.0
        prev_f2 = ref2
        for f1, f2 in pts:
            if f2 < prev_f2:
                hv += (ref1 - f1) * (prev_f2 - f2)
                prev_f2 = f2
        return hv

    def evaluate_and_insert(bits):
        nonlocal prune_thr
        # normalize
        bl = (bits + [0]*n_bits)[:n_bits]
        # cheap pruning by conflict density
        cd = conflict_density(bl)
        if cd > prune_thr and random.random() < 0.85:
            return False
        # repair geometry quickly
        if cd > 0.0:
            bl = repair_planarity(bl)
        # light connectivity enforcement for stiffness potential
        try:
            bl = ensure_connectivity(bl, max_add_edges=3)
        except Exception:
            pass
        if conflict_density(bl) > 0.0:
            bl = repair_planarity(bl)

        # final feasibility via API trimming
        if truss_problem.evaluate_constraints(bl) != 0:
            r = bl[:]
            ones = [i for i,v in enumerate(r) if v]
            ones.sort(key=lambda x: edge_len[x], reverse=True)
            for idx_rm in ones:
                r[idx_rm] = 0
                if truss_problem.evaluate_constraints(r) == 0:
                    break
            bl = r
        if truss_problem.evaluate_constraints(bl) != 0:
            return False

        # adaptive surrogate gating to save evaluations
        archive_size = len(archive_bits)
        elapsed = time.time() - t0_search
        time_frac = elapsed / max(1.0, ALGORITHM_TIMEOUT_SECONDS)
        surr_min = SURROGATE_BASE_MIN + 0.005 * min(archive_size, 20) + 0.08 * max(0.0, time_frac - 0.7)
        surr_skip = min(0.85, SURROGATE_SKIP_PROB + 0.01 * max(0, archive_size - 6))
        surrogate_score = 0.0
        for i, v in enumerate(bl):
            if v:
                surrogate_score += (base_weight[i] + 0.6 * edge_score[i])
        # Diversity-aware adjustment: novel structures get relaxed gating
        nov = novelty_score_bits(bl)
        surr_min_adj = surr_min * (1.0 - 0.25 * nov)
        surr_skip_adj = max(0.15, surr_skip * (1.0 - 0.5 * nov))
        if surrogate_score < surr_min_adj and random.random() < surr_skip_adj:
            return False

        key = tuple(int(x) for x in bl)
        if key in eval_cache:
            neg_s, vol = eval_cache[key]
        else:
            neg_s, vol = truss_problem.evaluate(bl)
            eval_cache[key] = (neg_s, vol)
        bstr = bitlist_to_str(bl)
        if bstr in seen:
            return False
        inserted = pareto_insert_grid(bl, (neg_s, vol), archive_objs, archive_bits)
        update_prune(inserted)
        if inserted:
            seen.add(bstr)
            for i,v in enumerate(bl):
                if v: edge_score[i] += 1.0
            # HV-based intensification: if HV improved, explore neighborhood
            try:
                hv_val = hv_of_front(archive_objs)
                if hv_val > BEST_HV[0] + 1e-9:
                    BEST_HV[0] = hv_val
                    LAST_HV_IMPROVE[0] = time.time()
                    prune_thr = max(0.32, prune_thr - 0.02)
                    # quick local neighborhood: densify and global mutate+repair
                    child = densify_planar(bl, target_additions=1)
                    evaluate_and_insert(child)
                    mutated = mutate_repair(bl, n_bits, truss_problem, k_flips=max(1, int(0.01 * n_bits)))
                    if mutated is not None:
                        evaluate_and_insert(mutated)
                else:
                    if time.time() - LAST_HV_IMPROVE[0] > 8.0:
                        prune_thr = min(0.62, prune_thr + 0.03)
            except Exception:
                pass
            last_improve[0] = time.time()
            return True
        else:
            try:
                reservoir_add(bl, (neg_s, vol))
            except Exception:
                pass
            return False

    # ---------- Seeding ----------
    # Early seeds: greedy planar, star seeds, geometry-informed seeds
    for ratio in (0.06, 0.12):
        t_edges = max(1, int(n_bits*ratio))
        evaluate_and_insert(greedy_planar_seed(t_edges))
        evaluate_and_insert(densify_planar(greedy_planar_seed(t_edges), target_additions=2))
    for ln in load_node_idx:
        evaluate_and_insert(star_seed(ln, 3))
        evaluate_and_insert(star_seed(ln, 5))
    # convex hull + boundary ring + bridge
    try:
        evaluate_and_insert(seed_convex_hull())
        evaluate_and_insert(densify_planar(seed_convex_hull(), target_additions=2))
    except Exception:
        pass
    try:
        evaluate_and_insert(seed_boundary_ring())
    except Exception:
        pass
    try:
        evaluate_and_insert(bridge_seed())
    except Exception:
        pass

    # quick save
    if archive_bits:
        try:
            truss_problem.save_solutions(np.array(archive_bits, dtype=int))
            last_save = time.time()
        except Exception:
            pass

    # ---------- Main exploration loop ----------
    # iterate combinations of static/free node subsets to generate NodeSort seeds
    for static_nodes_comb in static_nodes_comb_feasible:
        if time.time() - t0_search > ALGORITHM_TIMEOUT_SECONDS - 1.0:
            break
        for free_node_comb in free_nodes_pwr_set:
            if time.time() - t0_search > ALGORITHM_TIMEOUT_SECONDS - 1.0:
                break

            node_comb = list(set(list(free_node_comb) + list(static_nodes_comb)))
            nodes_new = [nodes[idx] for idx in node_comb]
            edges_new = node_sort_truss(nodes_new)
            bit_list, bit_str, node_idx_pairs, node_coords = truss_problem.convert(edges_new)
            node_idx_pairs_fixed = remove_members_connecting_fixed(node_idx_pairs, fixed_node_idx)
            bit_list, bit_str, node_idx_pairs, node_coords = truss_problem.convert(node_idx_pairs_fixed)

            # duplicates: reseed/intensify on stagnation
            if bit_str in seen:
                if time.time() - last_improve[0] > STAG_SEC:
                    bl1 = greedy_planar_seed(max(1, int(0.08*n_bits)))
                    bl2 = seed_boundary_ring()
                    evaluate_and_insert(densify_planar(bl1, target_additions=2))
                    evaluate_and_insert(densify_planar(bl2, target_additions=2))
                    if archive_bits:
                        leaders = random.sample(archive_bits, min(len(archive_bits), 3))
                        frontier_local_refine(leaders, budget_iters=5)
                    last_improve[0] = time.time()
                if time.time() - last_save > 1.3:
                    if archive_bits:
                        truss_problem.save_solutions(np.array(archive_bits, dtype=int))
                        last_save = time.time()
                continue

            # adaptive pruning on conflict density
            if conflict_density(bit_list) > prune_thr and random.random() < 0.80:
                continue

            # quick repair + evaluate seed + densifications
            bit_list = repair_planarity(bit_list)
            evaluate_and_insert(bit_list)
            evaluate_and_insert(densify_planar(bit_list, target_additions=2))

            # small local sweep focused on seed
            def local_seed_sweep(seed_bits, max_iters=6):
                cur = seed_bits[:]
                for it in range(max_iters):
                    k_try = max(1, int((0.008 if it%3 else 0.02)*n_bits))
                    nb = localized_mutation(cur, k_try)
                    if nb is None: continue
                    if evaluate_and_insert(nb):
                        cur = nb
            try:
                local_seed_sweep(bit_list, max_iters=6)
            except Exception:
                pass

            # occasional neighborhood mutations (localized + global)
            for k_mut in (max(1, int(0.005*n_bits)), max(1, int(0.02*n_bits))):
                if time.time() - t0_search > ALGORITHM_TIMEOUT_SECONDS - 1.0:
                    break
                mutated = localized_mutation(bit_list, k_mut)
                evaluate_and_insert(mutated)
            # one global mutate+repair attempt
            gm = mutate_repair(bit_list, n_bits, truss_problem, k_flips=max(1, int(0.015*n_bits)))
            if gm is not None:
                evaluate_and_insert(gm)

            # periodic save
            if time.time() - last_save > 1.3:
                if archive_bits:
                    truss_problem.save_solutions(np.array(archive_bits, dtype=int))
                last_save = time.time()

    # ---------- Frontier refinement (final phase) ----------
    t_end = t0_search + (ALGORITHM_TIMEOUT_SECONDS - 0.5)
    while time.time() < t_end:
        if not archive_bits and not reservoir:
            break
        # choose parent from archive majorly, occasionally reservoir
        if archive_bits and random.random() < 0.75:
            parent = random.choice(archive_bits)
        else:
            sample = reservoir_sample()
            parent = sample if sample is not None else (random.choice(archive_bits) if archive_bits else None)
        if parent is None: break
        r = random.random()
        if r < 0.45:
            child = densify_planar(parent, target_additions=2)
            evaluate_and_insert(child)
        elif r < 0.85:
            k_mut = max(1, int(0.01*n_bits))
            child = localized_mutation(parent, k_mut)
            evaluate_and_insert(child)
            if random.random() < 0.4:
                evaluate_and_insert(densify_planar(child, target_additions=1))
        else:
            leaders = random.sample(archive_bits, min(len(archive_bits), 4))
            frontier_local_refine(leaders, budget_iters=4)
        # try a global mutate+repair occasionally
        if random.random() < 0.25:
            gm2 = mutate_repair(parent, n_bits, truss_problem, k_flips=max(1, int(0.012*n_bits)))
            if gm2 is not None:
                evaluate_and_insert(gm2)
        # periodic saves and light decay of edge_score
        if time.time() - last_save > 1.0:
            if archive_bits:
                truss_problem.save_solutions(np.array(archive_bits, dtype=int))
            edge_score[:] = [s*0.93 for s in edge_score]
            last_save = time.time()

    # final save
    if archive_bits:
        truss_problem.save_solutions(np.array(archive_bits))
    else:
        truss_problem.save_solutions(np.array([]))

# ---------- Auxiliary functions (outside search_algorithm) ----------
def node_sort_truss(nodes):
    sorted_nodes = sorted(nodes)
    n = len(sorted_nodes)
    edges = []
    for i in range(n-1):
        node_i = sorted_nodes[i]
        next_node_index = i+1
        if next_node_index >= n: break
        node_next = sorted_nodes[next_node_index]
        targets = []
        if node_i[1] >= node_next[1]:
            above_count = 0
            for j in range(i+1, n):
                candidate = sorted_nodes[j]
                if targets:
                    prev = targets[-1]
                    if candidate[1] < prev[1]: break
                is_above = candidate[1] > node_i[1]
                if is_above: above_count += 1
                targets.append(candidate)
                if above_count == 2: break
        else:
            below_count = 0
            for j in range(i+1, n):
                candidate = sorted_nodes[j]
                if targets:
                    prev = targets[-1]
                    if candidate[1] > prev[1]: break
                is_below = candidate[1] < node_i[1]
                if is_below: below_count += 1
                targets.append(candidate)
                if below_count == 2: break
        for t in targets:
            edges.append((node_i, t))
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

    save_dir = '/Users/gapaza/repos/ideal/combench/plots/shinka/alg5'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    pop.plot_population(save_dir)









