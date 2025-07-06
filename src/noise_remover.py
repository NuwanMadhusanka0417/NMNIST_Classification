import numpy as np
from typing import Tuple

def purge_sparse_voxels(
    events: np.ndarray,
    *,
    sensor_size: Tuple[int, int] = (34, 34),   # (W, H)
    xy_bin_size: int = 5,                      # spatial voxel = 5×5 px
    time_bin_us: int = 20_000,                 # 20 ms
    min_events: int = 10                       # threshold: keep voxels ≥ 10 events
) -> np.ndarray:
    """
    Remove events that lie in voxels containing < min_events events.

    Parameters
    ----------
    events : structured array with fields 'x','y','t' (and optionally 'p')
    sensor_size : full sensor resolution (W, H)
    xy_bin_size : bin width for x & y in pixels
    time_bin_us : temporal bin width in micro-seconds
    min_events  : voxel size threshold

    Returns
    -------
    filtered_events : same dtype as `events`, comprising only events in dense voxels
    """
    assert {"x", "y", "t"}.issubset(events.dtype.names), "events must have x, y, t fields"

    W, H = sensor_size
    n_x_bins = (W + xy_bin_size - 1) // xy_bin_size    # ceil
    n_y_bins = (H + xy_bin_size - 1) // xy_bin_size

    # ── 1. voxel indices for every event ────────────────────────────────────────
    x_idx = np.clip(events["x"] // xy_bin_size, 0, n_x_bins - 1)
    y_idx = np.clip(events["y"] // xy_bin_size, 0, n_y_bins - 1)
    t_idx = (events["t"] // time_bin_us).astype(np.int64)     # starts at 0

    flat_idx = (t_idx * n_y_bins * n_x_bins) + (y_idx * n_x_bins) + x_idx

    # ── 2. count events per voxel ──────────────────────────────────────────────
    counts_flat = np.bincount(flat_idx, minlength=flat_idx.max() + 1)

    # ── 3. mask: keep events whose voxel count ≥ threshold ─────────────────────
    keep_mask = counts_flat[flat_idx] >= min_events

    return events[keep_mask]