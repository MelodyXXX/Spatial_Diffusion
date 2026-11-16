# connectivity.py
"""
Room-level Connectivity F1 shared by Fig14_cal_metrics.py, FigS2_complexity_stratified.py
and ablation_smoothing.py.

Definition:
  - Rooms are keyed by (floor, type, type_id); type_id is numbered per floor, so
    the floor is part of a room's identity (as in GraphConstructor.get_program_unicode).
  - Predicted edges come from voxel adjacency. Across floors only stairs-stairs and
    elevator-elevator links are kept, because the local graph (GT) links floors only
    through vertical circulation; any other pair of stacked voxels is not an edge.
  - Predicted room instances are matched to GT rooms by maximum voxel overlap
    (Hungarian assignment per floor x type) before comparing edges, so the score
    does not depend on how instance ids happen to be numbered. Predicted instances
    left unmatched get ids absent from the GT, so their edges count as false positives.
  - Diffusion outputs are re-segmented into instances from their per-voxel classes
    with the inference-time recovery (get_program_metrics_and_instances, horizontal
    plane fixed to the x/y axes); Building-GAN predicts room assignments directly
    and keeps its own instances.
"""
import json
from collections import defaultdict

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch_geometric.data import Batch

from utils.eval.evaldi import get_program_metrics_and_instances, assign_voxel_instance_ids
from utils.eval.metrics import calculate_connectivity_metrics

VERTICAL_TYPES = {2, 3}    # stairs, elevator
FOLLOW_BATCH = ["program_target_ratio", "program_class_feature", "voxel_feature"]  # as in inference
UNMATCHED_ID_BASE = 10000  # ids for predicted instances with no GT counterpart


def _floor(n):
    return int(round((n.get("location") or n.get("coordinate", [0, 0, 0]))[0]))


def voxel_room_edges(voxel_graph, floor_aware=True, aligned=True):
    """Room-pair edges from voxel adjacency. floor_aware=False, aligned=False reproduces the
    original (type, type_id) definition."""
    nodes = voxel_graph["voxel_node"]
    loc_to_idx = {tuple(int(round(x)) for x in (n.get("location") or n.get("coordinate", [0, 0, 0]))): i
                  for i, n in enumerate(nodes)}

    def key(n):
        k = (int(n["type"]), int(n["type_id"]))
        return (_floor(n),) + k if floor_aware else k

    edges = set()
    for n in nodes:
        t = int(n["type"])
        if t == -1:
            continue
        u = key(n)
        for nb_raw in n.get("neighbors", []):
            j = loc_to_idx.get(tuple(int(round(x)) for x in nb_raw))
            if j is None or int(nodes[j]["type"]) == -1:
                continue
            m = nodes[j]
            if aligned and _floor(m) != _floor(n) and not (int(m["type"]) == t and t in VERTICAL_TYPES):
                continue
            v = key(m)
            if u != v:
                edges.add(tuple(sorted((u, v))))
    return edges


def local_room_edges(local_graph, floor_aware=True):
    nodes = local_graph["node"]
    key = (lambda f, t, i: (f, t, i)) if floor_aware else (lambda f, t, i: (t, i))
    valid = {key(n["floor"], n["type"], n["type_id"]) for n in nodes}
    edges = set()
    for n in nodes:
        u = key(n["floor"], n["type"], n["type_id"])
        for nb in n["neighbors"]:
            v = key(nb[0], nb[1], nb[2])
            if v in valid and u != v:
                edges.add(tuple(sorted((u, v))))
    return edges


def recover_instances(pt, voxel_graph, local_graph, num_classes):
    """Re-segments a voxel graph's per-voxel classes into room instances with the
    inference-time recovery. pt is the sample's graph (same voxel order as voxel_graph)."""
    labels = torch.tensor([int(n["type"]) for n in voxel_graph["voxel_node"]], dtype=torch.long)
    batch = Batch.from_data_list([pt], follow_batch=FOLLOW_BATCH)
    _, _, _, voxel_pid = get_program_metrics_and_instances(batch, labels, num_classes)
    out = json.loads(json.dumps(voxel_graph))
    lookup = {idx: [n["type"], n["type_id"]] for idx, n in enumerate(local_graph["node"])}
    assign_voxel_instance_ids(out, labels.numpy(), voxel_pid, 0, lookup)
    return out


def match_instances_to_gt(pred_graph, gt_graph, local_graph):
    """Renames predicted instance ids to the GT rooms they overlap most (Hungarian, per floor x
    type). Both voxel graphs share voxel order. Returns a new voxel graph."""
    pred_nodes, gt_nodes = pred_graph["voxel_node"], gt_graph["voxel_node"]
    overlap = defaultdict(lambda: defaultdict(int))   # (floor, type) -> {(pred_id, gt_id): count}
    pred_ids = defaultdict(set)
    for p, g in zip(pred_nodes, gt_nodes):
        t = int(p["type"])
        if t == -1:
            continue
        grp = (_floor(p), t)
        pred_ids[grp].add(int(p["type_id"]))
        if int(g["type"]) == t:
            overlap[grp][(int(p["type_id"]), int(g["type_id"]))] += 1

    gt_ids = defaultdict(set)
    for n in local_graph["node"]:
        gt_ids[(n["floor"], n["type"])].add(n["type_id"])

    rename = {}
    for grp, pids in pred_ids.items():
        P, G = sorted(pids), sorted(gt_ids.get(grp, ()))
        matched = set()
        if G:
            W = np.array([[overlap[grp][(pi, gi)] for gi in G] for pi in P], dtype=float)
            for r, c in zip(*linear_sum_assignment(-W)):
                rename[grp + (P[r],)] = G[c]
                matched.add(P[r])
        for k, pi in enumerate(x for x in P if x not in matched):
            rename[grp + (pi,)] = UNMATCHED_ID_BASE + k

    out = json.loads(json.dumps(pred_graph))
    for n in out["voxel_node"]:
        if int(n["type"]) != -1:
            n["type_id"] = rename[(_floor(n), int(n["type"]), int(n["type_id"]))]
    return out


def prepare_prediction(pred_graph, gt_graph, local_graph, pt, num_classes, rerecover):
    """Prediction voxel graph with instances as evaluated: re-segmented (diffusion) and matched to GT."""
    if rerecover:
        pred_graph = recover_instances(pt, pred_graph, local_graph, num_classes)
    return match_instances_to_gt(pred_graph, gt_graph, local_graph)


def connectivity_scores(pred_graph, local_graph):
    """(precision, recall, F1) of a prepared prediction against the GT local graph."""
    return calculate_connectivity_metrics(voxel_room_edges(pred_graph), local_room_edges(local_graph))
