import os
import numpy as np
from scipy.io import loadmat

# ---------- file reader (ASL-DVS) ----------
def read_asldvs_mat_int(path):
    """
    Read one ASL-DVS .mat file and return a structured array:
    dtype = [('t','i4'), ('x','i2'), ('y','i2'), ('p','i2')]

    Expected keys in .mat:
      - 'ts'  : timestamps
      - 'x'   : x coords
      - 'y'   : y coords
      - 'pol' : polarity

    NOTE: If your timestamps can exceed int32, switch 'i4' -> 'i8'.
    """
    mat = loadmat(path)

    # Try direct keys first
    if all(k in mat for k in ('ts', 'x', 'y', 'pol')):
        t  = np.ravel(mat['ts'])
        x  = np.ravel(mat['x'])
        y  = np.ravel(mat['y'])
        p  = np.ravel(mat['pol'])
    else:
        # Fallback: find the first non-dunder key that looks like a struct/dict
        data_key = next((k for k in mat.keys() if not k.startswith("__")), None)
        if data_key is None:
            raise KeyError(f"No usable data found in {path}")

        d = mat[data_key]
        try:
            t = np.ravel(np.array(d['ts']))
            x = np.ravel(np.array(d['x']))
            y = np.ravel(np.array(d['y']))
            p = np.ravel(np.array(d['pol']))
        except Exception as e:
            raise KeyError(f"Could not locate ts/x/y/pol in {path}: {e}")

    # Dtypes (match your SNKTH style)
    t = np.rint(t).astype(np.int32)   # use .astype(np.int64) if overflow is possible
    x = x.astype(np.int16)
    y = y.astype(np.int16)
    p = p.astype(np.int16)            # keep as {-1,1} or {0,1} as stored

    dtype = [('t', 'i4'), ('x', 'i2'), ('y', 'i2'), ('p', 'i2')]
    result = np.empty(t.shape[0], dtype=dtype)
    result['t'] = t
    result['x'] = x
    result['y'] = y
    result['p'] = p
    return result


# ---------- dataset class (ASL-DVS) ----------
class ASLDVS:
    def __init__(self, root):
        """
        Initialize ASLDVS by collecting all .mat files and their labels.
        Assumes folder structure like:
            root/
              A/ *.mat
              B/ *.mat
              ...
        (class names are auto-discovered from subfolders)
        """
        self.files = []
        # discover class folders automatically (sorted for stable label ids)
        classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        if not classes:
            raise RuntimeError(f"No class subfolders found under {root}")

        self.class_to_label = {c: i for i, c in enumerate(classes)}
        self.label_to_class = {i: c for c, i in self.class_to_label.items()}

        for c in classes:
            cdir = os.path.join(root, c)
            for fname in os.listdir(cdir):
                if fname.lower().endswith(".mat"):
                    self.files.append((os.path.join(cdir, fname), self.class_to_label[c]))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, label = self.files[idx]
        data = read_asldvs_mat_int(path)
        return data, label
