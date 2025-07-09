import numpy as np
from scipy.spatial import cKDTree
import torch
from torch_geometric.data import Data
from collections import defaultdict

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data

def events_to_st_graph(
        events: np.ndarray,
        *,
        sensor_size=(34, 34),     # (W, H) for normalising features
        beta: float = 1e-4,       # time normalization factor
        R: float = 4.0,           # spatio-temporal radius threshold
        Dmax: int = 18,           # max neighbors per node
        directed: bool = True,   # if True: only edges i→j where t_i <= t_j
        normalized_feat: bool = False
) -> Data:
    """
    Build a graph where nodes = events and edges connect all neighbours
    within a 3D radius R in (x, y, beta*t) space, capped to Dmax per node.

    Parameters
    ----------
    events        : structured array with fields 'x','y','t' (plus optional 'p')
    sensor_size   : to normalize x,y into [0,1]
    beta          : scale factor for timestamps
    R             : max Euclidean distance in (x,y,beta*t) for connectivity
    Dmax          : maximum number of neighbors per node
    directed      : if True, only link i→j when t_i <= t_j
    normalized_feat : whether to normalize features

    Returns
    -------
    Data  with fields
      x           node features  (N,3 or 4)
      edge_index  (2, E)
    """

    # print("LOG - R=",R, " | Dmax=", Dmax)
    assert {"x", "y", "t"}.issubset(events.dtype.names)
    N = len(events)
    # 1) build 3D coords
    xs = events["x"].astype(np.float32)
    ys = events["y"].astype(np.float32)
    ts = events["t"].astype(np.float32)
    t_star = beta * ts
    coords3 = np.stack([xs, ys, t_star], axis=1)

    tree = cKDTree(coords3)
    # for each point, find all within R (including itself)
    neighbors = tree.query_ball_point(coords3, r=R)

    src, dst = [], []
    for i, nbrs in enumerate(neighbors):
        # remove self
        nbrs = [j for j in nbrs if j != i]
        if not nbrs:
            continue
        # compute distances
        dists = np.linalg.norm(coords3[nbrs] - coords3[i], axis=1)
        # sort by distance
        sorted_idx = np.argsort(dists)
        # keep up to Dmax closest
        for idx in sorted_idx[:Dmax]:
            j = nbrs[idx]
            # directed option
            if directed and ts[j] < ts[i]:
                continue
            src.append(i)
            dst.append(j)
            if not directed:
                src.append(j)
                dst.append(i)

    if len(src) == 0:
        edge_index = np.empty((2, 0), dtype=np.int64)
    else:
        edge_index = np.vstack([src, dst])

    # 2) node features
    if normalized_feat:
        W, H = sensor_size
        x_n  = xs / W
        y_n  = ys / H
        t_n  = ts - ts.min()  # keep raw scale in ms*beta
        if "p" in events.dtype.names:
            p_n = events["p"].astype(np.float32) * 2 - 1
            feats = np.stack([x_n, y_n, t_n, p_n], axis=1)
        else:
            feats = np.stack([x_n, y_n, t_n], axis=1)
    else:
        if "p" in events.dtype.names:
            p_n = events["p"].astype(np.float32)
            feats = np.stack([xs, ys, ts, p_n], axis=1)
        else:
            feats = np.stack([xs, ys, ts], axis=1)

    return Data(
        x=torch.tensor(feats, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long)
    )

### method that use only next node and same pixel node as neighbours.
def events_to_sota_graph(
    events: np.ndarray,
    *,
    sensor_size=(34, 34),
    spatial_thr: float = 3.0,  # unused now, kept for signature
    time_thr_us: int = 100,    # unused now, kept for signature
    directed: bool = False,    # unused now, kept for signature
    normalized_feat: bool = True,
    tau: int = 2,              # new temporal degree
    fs=None,
    ft=None
) -> Data:
    """
    Build a spatio-temporal event graph as torch_geometric.data.Data

    Nodes: each event ε_i = (x, y, t, p)
    Edges:
      • next-in-time: i-1 → i with (α,β)
      • same-pixel: up to tau successors at same (x,y), α=0

    Returns Data(x, edge_index, edge_attr)
    """
    # assert {"x", "y", "t"}.issubset(events.dtype.names), "events must have x, y, t"
    # 1) Prepare node features
    coords = events  # assume events fields: x,y,t[,p]
    xs, ys, ts = coords["x"], coords["y"], coords["t"]
    if "p" in coords.dtype.names:
        ps = coords["p"] * 2 - 1 if normalized_feat else coords["p"]
        feats = np.stack([xs, ys, ts - ts.min(), ps], axis=1)
    else:
        feats = np.stack([xs, ys, ts - ts.min()], axis=1)

    x = torch.tensor(feats, dtype=torch.float32)

    # 2) distance functions
    if fs is None:
        fs = lambda s1, s2: np.hypot(s1[0] - s2[0], s1[1] - s2[1])
    if ft is None:
        ft = lambda t2, t1: abs(t2 - t1)

    # 3) Build edges
    N = len(events)
    row, col, edge_attrs = [], [], []

    # next-in-time
    for i in range(1, N):
        j = i - 1
        α = fs((xs[i], ys[i]), (xs[j], ys[j]))
        β = ft(ts[i], ts[j])
        row.append(j); col.append(i)
        edge_attrs.append([α, β])

    # same-pixel up to tau
    by_pix = defaultdict(list)
    for i, (xi, yi) in enumerate(zip(xs, ys)):
        by_pix[(int(xi), int(yi))].append(i)
    for idxs in by_pix.values():
        for k, i in enumerate(idxs):
            for d in range(1, tau + 1):
                if k + d < len(idxs):
                    j = idxs[k + d]
                    row.append(i); col.append(j)
                    edge_attrs.append([0.0, ft(ts[j], ts[i])])
                else:
                    break

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr  = torch.tensor(edge_attrs, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)