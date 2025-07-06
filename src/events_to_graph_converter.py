import numpy as np
from scipy.spatial import cKDTree
import torch
from torch_geometric.data import Data
from collections import defaultdict

def events_to_st_graph(
        events: np.ndarray,
        *,
        sensor_size=(34, 34),     # (W, H) for normalising features
        spatial_thr: float = 3.0, # pixel radius
        time_thr_us: int = 100,   # µs
        directed: bool = False,    # if True: only t_i < t_j edges
        normalized_feat:bool = True
) -> Data:
    """
    Build a graph where nodes = events and edges connect spatio-temporal neighbours.

    Parameters
    ----------
    events        : structured array with fields 'x','y','t' (plus optional 'p')
    spatial_thr   : max spatial distance in pixels
    time_thr_us   : max |Δt| in micro-seconds
    directed      : True → edges i→j only if t_i <  t_j  and Δt <= time_thr_us
                    False → undirected edges if |Δt| <= time_thr_us

    Returns
    -------
    torch_geometric.data.Data  with fields
        x           node features  (N, 3|4)
        edge_index  (2, E) tensor  (undirected or directed as requested)
    """
    assert {"x", "y", "t"}.issubset(events.dtype.names), "events must have x, y, t"

    # ── 1. KD-tree on spatial coords ────────────────────────────────────────────
    coords = np.stack([events["x"], events["y"]], axis=1).astype(np.float32)
    tree   = cKDTree(coords)
    pairs  = np.array(list(tree.query_pairs(r=spatial_thr, output_type="set")),
                      dtype=np.int64)             # shape (P,2) with i<j
    if pairs.size == 0:                           # no spatial neighbours
        edge_index = np.empty((2, 0), dtype=np.int64)
    else:
        ti, tj = events["t"][pairs[:, 0]], events["t"][pairs[:, 1]]
        dt     = tj - ti

        if directed:
            keep = (dt > 0) & (dt <= time_thr_us)
            edge_index = pairs[keep].T           # i→j forward only
        else:
            keep = np.abs(dt) <= time_thr_us
            undirected = pairs[keep]
            edge_index = np.concatenate([undirected.T, undirected.T[::-1]], axis=1)

    # ── 2. Node feature matrix  -------------------------------------------------
    if normalized_feat:
        W, H = sensor_size
        x_n  = events["x"] / W             # x and y doesn't normalize. this will affect on GNN, but need to do this for codebook
        y_n  = events["y"] / H
        t_n  = (events["t"] - events["t"].min()) # / max(1, events["t"].ptp())  # 0-1  # time is very importatnt to identify running, walking, jogging etc. so normalization reduce information.
        p_n = events["p"].astype(np.float32) * 2 - 1          # 0→-1, 1→+1
    else:
        x_n = events["x"]               # x and y doesn't normalize. this will affect on GNN, but need to do this for codebook
        y_n = events["y"]
        t_n = (events["t"] - events["t"].min())   / max(1, events["t"].ptp())  # 0-1  # time is very importatnt to identify running, walking, jogging etc. so normalization reduce information.
        p_n = events["p"]

    if "p" in events.dtype.names:
        feats = np.stack([x_n, y_n, t_n, p_n], axis=1)
    else:
        feats = np.stack([x_n, y_n, t_n], axis=1)

    return Data(
        x=torch.tensor(feats, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long)
    )

# === your updated graph‐building function ===
def events_to_sota_graph_________________(events, tau=2, fs=None, ft=None):
    import numpy as np
    import torch
    from torch_geometric.data import Data
    from collections import defaultdict

    if fs is None:
        fs = lambda s1, s2: np.hypot(s1[0] - s2[0], s1[1] - s2[1])
    if ft is None:
        ft = lambda t2, t1: abs(t2 - t1)

    ev = np.asarray(events, dtype=float)
    xs, ys, ts, ps = ev[:,0], ev[:,1], ev[:,2], ev[:,3]
    N = len(ev)

    # Node features [x, y, t, p]
    x = torch.tensor(ev, dtype=torch.float)

    row, col, edge_attrs = [], [], []
    # next‐in‐time edges
    for i in range(1, N):
        j = i - 1
        row.append(j); col.append(i)
        edge_attrs.append([fs((xs[i],ys[i]), (xs[j],ys[j])), ft(ts[i], ts[j])])
    # same‐pixel edges
    by_pix = defaultdict(list)
    for idx, (xi, yi) in enumerate(zip(xs, ys)):
        by_pix[(int(xi),int(yi))].append(idx)
    for idxs in by_pix.values():
        for k, i in enumerate(idxs):
            for d in range(1, tau+1):
                if k+d < len(idxs):
                    j = idxs[k+d]
                    row.append(i); col.append(j)
                    edge_attrs.append([0.0, ft(ts[j], ts[i])])
                else:
                    break

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr  = torch.tensor(edge_attrs, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


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