import os
import numpy as np
from scipy.io import loadmat

import numpy as np
from scipy.io import loadmat

def read_snkth_mat_int(path):
    mat_data = loadmat(path)

    # Automatically find the relevant data key
    data_key = next((k for k in mat_data.keys() if not k.startswith("__")), None)
    if data_key is None:
        raise KeyError(f"No usable data found in {path}")

    raw = mat_data[data_key]  # e.g., shape (N, 4) or structured array

    # Check if it's packed event data (assumed 4 columns: t, _, ...), or already split
    if raw.shape[1] == 4 and isinstance(raw, np.ndarray):
        # Example: t, _, other1, other2
        packed = raw[:, 1].astype(np.int32)  # assuming 2nd col is "_"

        x = np.bitwise_and(packed, 16383).astype(np.int16)
        y = np.right_shift(np.bitwise_and(packed, 268419072), 14).astype(np.int16)
        p = np.right_shift(np.bitwise_and(packed, 268435456), 28).astype(np.int16)

        dtype = [('t', 'i4'), ('x', 'i2'), ('y', 'i2'), ('p', 'i2')]
        result = np.empty(len(packed), dtype=dtype)
        result['t'] = raw[:, 0].astype(np.int32)
        result['x'] = x
        result['y'] = y
        result['p'] = p
        return result

    elif {'x', 'y', 'p', 't'}.issubset(set(mat_data.keys())):
        # already unpacked fields
        x = mat_data['x'].squeeze().astype(np.int16)
        y = mat_data['y'].squeeze().astype(np.int16)
        p = mat_data['p'].squeeze().astype(np.int16)
        t = mat_data['t'].squeeze().astype(np.int32)

        dtype = [('t', 'i4'), ('x', 'i2'), ('y', 'i2'), ('p', 'i2')]
        result = np.empty(len(t), dtype=dtype)
        result['t'] = t
        result['x'] = x
        result['y'] = y
        result['p'] = p
        return result

    else:
        raise ValueError("Unknown structure of .mat file; expected packed or x/y/p/t fields")


class SNKTH():
    def __init__(self, root, split="train"):
        """
        Initialize SNKTH dataset by collecting all .mat files and their labels.
        Assumes `root/split/` contains subfolders for each class, similar to NCARS.
        """
        self.files = []
        # Replace ["classA", "classB", ...] with actual class folder names in SNKTH dataset:
        class_folders = ["jogging", "running","boxing","handclapping","handwaving","walking"]  # example class names
        for label, sub in enumerate(class_folders):
            folder = os.path.join(root,  sub)
            for fname in os.listdir(folder):
                if fname.lower().endswith(".mat"):
                    filepath = os.path.join(folder, fname)
                    self.files.append((filepath, label))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, label = self.files[idx]
        data = read_snkth_mat_int(path)   # load the .mat file data (events or features)
        return data, label
