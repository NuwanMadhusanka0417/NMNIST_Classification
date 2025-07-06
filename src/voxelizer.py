import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, Any

def voxelize_events(
    events: np.ndarray,
    sensor_size: Tuple[int, int] = (34, 34),   # (W, H)
    xy_bin_size: int = 5,                      # 5 px → 7 bins along each axis
    time_bin_us: int = 20_000                  # 20 ms  (micro-seconds)
) -> Any:
    """
    Parameters
    ----------
    events : structured NumPy array with fields 'x', 'y', 't', 'p' (polarity optional)
             x,y ∈ [0,33],  t in µs
    sensor_size : (W,H) full resolution of the sensor
    xy_bin_size : spatial bin width in pixels
    time_bin_us : temporal bin width in micro-seconds

    Returns
    -------
    voxels : dict  { (t_bin, y_bin, x_bin) : view_of_events }
             empty voxels do not appear in the dict
    """
    assert {"x", "y", "t"}.issubset(events.dtype.names), "events must have x, y, t fields"

    W, H = sensor_size
    n_x_bins = (W + xy_bin_size - 1) // xy_bin_size        # ceil
    n_y_bins = (H + xy_bin_size - 1) // xy_bin_size

    # -------- 1. compute voxel indices for every event -------------------------
    x_idx = np.clip(events["x"] // xy_bin_size, 0, n_x_bins - 1)
    y_idx = np.clip(events["y"] // xy_bin_size, 0, n_y_bins - 1)
    t_idx = (events["t"] // time_bin_us).astype(np.int64)  # start at 0

    # -------- 2. group events by (t_bin, y_bin, x_bin) -------------------------
    voxels = defaultdict(list)
    for i, key in enumerate(zip(t_idx, y_idx, x_idx)):
        voxels[key].append(i)               # store indices to avoid copies

    # -------- 3. convert lists of indices into *views* of the event packet -----
    for key, idx_list in voxels.items():
        voxels[key] = events[np.array(idx_list)]

    '''
    Ig dence tensor needed
    out = np.empty((T, n_h_bins, n_w_bins), dtype=object)   # each cell holds a view
    for key, idx_list in voxels.items():
        t, h, w = key
        out[t, h, w] = events[np.array(idx_list)]
    '''

    flat_idx = (t_idx   * n_y_bins * n_x_bins) + \
           (y_idx   * n_x_bins)            + x_idx

    counts = np.bincount(flat_idx, minlength=(t_idx.max()+1) * n_y_bins * n_x_bins).reshape(-1, n_y_bins, n_x_bins)

    return voxels, counts