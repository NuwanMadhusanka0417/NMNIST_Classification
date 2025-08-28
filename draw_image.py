import tonic
import tonic.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from src.loader import ev_loader
from torch.utils.data import Dataset, DataLoader
import os
import sys
from PIL import Image
EV_TYPE = [('t', 'u4'), ('_', 'i4')]  # Event2D

EV_STRING = 'Event2D'

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

class ASLDVS():
    def __init__(self, root, split="train"):
        """
        Initialize SNKTH dataset by collecting all .mat files and their labels.
        Assumes `root/split/` contains subfolders for each class, similar to NCARS.
        """
        self.files = []
        # Replace ["classA", "classB", ...] with actual class folder names in SNKTH dataset:
        class_folders = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y"]  # example class names
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
        data = read_asldvs_mat_int(path)   # load the .mat file data (events or features)
        return data, label
def load_td_data(filename, ev_count=-1, ev_start=0):
    """
    Loads TD data from files generated by the StreamLogger consumer for Event2D
    events [ts,x,y,p]. The type ID in the file header must be 0.
    args :
        - path to a dat file
        - number of event (all if set to the default -1)
        - index of the first event

    return :
        - dat, a dictionary like structure containing the fields ts, x, y, p
    """

    with open(filename, 'rb') as f:
        _, ev_type, ev_size, _ = parse_header(f)
        if ev_start > 0:
            f.seek(ev_start * ev_size, 1)

        dtype = EV_TYPE
        dat = np.fromfile(f, dtype=dtype, count=ev_count)
        xyp = None
        if ('_', 'i4') in dtype:
            x = np.bitwise_and(dat["_"], 16383)
            y = np.right_shift(
                np.bitwise_and(dat["_"], 268419072), 14)
            p = np.right_shift(np.bitwise_and(dat["_"], 268435456), 28)
            xyp = (x, y, p)
        return _dat_transfer(dat, dtype, xyp=xyp)


def _dat_transfer(dat, dtype, xyp=None):
    """
    Transfers the fields present in dtype from an old datastructure to a new datastructure
    xyp should be passed as a tuple
    args :
        - dat vector as directly read from file
        - dtype _numpy dtype_ as a list of couple of field name/ type eg [('x','i4'), ('y','f2')]
        - xyp optional tuple containing x,y,p extracted from a field '_'and untangled by bitshift and masking
    """
    variables = []
    xyp_index = -1
    for i, (name, _) in enumerate(dtype):
        if name == '_':
            xyp_index = i
            continue
        variables.append((name, dat[name]))
    if xyp and xyp_index == -1:
        print("Error dat didn't contain a '_' field !")
        return
    if xyp_index >= 0:
        dtype = dtype[:xyp_index] + [('x', 'i2'), ('y', 'i2'), ('p', 'i2')] + dtype[xyp_index + 1:]
    new_dat = np.empty(dat.shape[0], dtype=dtype)
    if xyp:
        new_dat["x"] = xyp[0].astype(np.uint16)
        new_dat["y"] = xyp[1].astype(np.uint16)
        new_dat["p"] = xyp[2].astype(np.uint16)
    for (name, arr) in variables:
        new_dat[name] = arr
    return new_dat


def stream_td_data(file_handle, buffer, dtype, ev_count=-1):
    """
    Streams data from opened file_handle
    args :
        - file_handle: file object
        - buffer: pre-allocated buffer to fill with events
        - dtype:  expected fields
        - ev_count: number of events
    """

    dat = np.fromfile(file_handle, dtype=dtype, count=ev_count)
    count = len(dat['t'])
    for name, _ in dtype:
        if name == '_':
            buffer['x'][:count] = np.bitwise_and(dat["_"], 16383)
            buffer['y'][:count] = np.right_shift(np.bitwise_and(dat["_"], 268419072), 14)
            buffer['p'][:count] = np.right_shift(np.bitwise_and(dat["_"], 268435456), 28)
        else:
            buffer[name][:count] = dat[name]


def count_events(filename):
    """
    Returns the number of events in a dat file
    args :
        - path to a dat file
    """
    with open(filename, 'rb') as f:
        bod, _, ev_size, _ = parse_header(f)
        f.seek(0, os.SEEK_END)
        eod = f.tell()
        if (eod - bod) % ev_size != 0:
            raise Exception("unexpected format !")
        return (eod - bod) // ev_size


def parse_header(f):
    """
    Parses the header of a dat file
    Args:
        - f file handle to a dat file
    return :
        - int position of the file cursor after the header
        - int type of event
        - int size of event in bytes
        - size (height, width) tuple of int or None
    """
    f.seek(0, os.SEEK_SET)
    bod = None
    end_of_header = False
    header = []
    num_comment_line = 0
    size = [None, None]
    # parse header
    while not end_of_header:
        bod = f.tell()
        line = f.readline()
        if sys.version_info > (3, 0):
            first_item = line.decode("latin-1")[:2]
        else:
            first_item = line[:2]

        if first_item != '% ':
            end_of_header = True
        else:
            words = line.split()
            if len(words) > 1:
                if words[1] == 'Date':
                    header += ['Date', words[2] + ' ' + words[3]]
                if words[1] == 'Height' or words[1] == b'Height':  # compliant with python 3 (and python2)
                    size[0] = int(words[2])
                    header += ['Height', words[2]]
                if words[1] == 'Width' or words[1] == b'Width':  # compliant with python 3 (and python2)
                    size[1] = int(words[2])
                    header += ['Width', words[2]]
            else:
                header += words[1:3]
            num_comment_line += 1
    # parse data
    f.seek(bod, os.SEEK_SET)

    if num_comment_line > 0:  # Ensure compatibility with previous files.
        # Read event type
        ev_type = np.frombuffer(f.read(1), dtype=np.uint8)[0]
        # Read event size
        ev_size = np.frombuffer(f.read(1), dtype=np.uint8)[0]
    else:
        ev_type = 0
        ev_size = sum([int(n[-1]) for _, n in EV_TYPE])

    bod = f.tell()
    return bod, ev_type, ev_size, size


def write_header(filename, height=240, width=320, ev_type=0):
    """
    write header for a dat file
    """
    if max(height, width) > 2**14 - 1:
        raise ValueError('Coordinates value exceed maximum range in'
                         ' binary .dat file format max({:d},{:d}) vs 2^14 - 1'.format(
                             height, width))
    f = open(filename, 'w')
    f.write('% Data file containing {:s} events.\n'
            '% Version 2\n'.format(EV_STRINGS[ev_type]))
    now = datetime.datetime.utcnow()
    f.write("% Date {}-{}-{} {}:{}:{}\n".format(now.year,
                                                now.month, now.day, now.hour,
                                                now.minute, now.second))

    f.write('% Height {:d}\n'
            '% Width {:d}\n'.format(height, width))
    # write type and bit size
    ev_size = sum([int(b[-1]) for _, b in EV_TYPE])

    np.array([ev_type, ev_size], dtype=np.uint8).tofile(f)
    f.flush()
    return f


def write_event_buffer(f, buffers):
    """
    writes events of fields x,y,p,t into the file object f
    """
    # pack data as events
    dtype = EV_TYPE
    data_to_write = np.empty(len(buffers['t']), dtype=dtype)

    for (name, typ) in buffers.dtype.fields.items():
        if name == 'x':
            x = buffers['x'].astype('i4')
        elif name == 'y':
            y = np.left_shift(buffers['y'].astype('i4'), 14)
        elif name == 'p':
            buffers['p'] = (buffers['p'] == 1).astype(buffers['p'].dtype)
            p = np.left_shift(buffers['p'].astype("i4"), 28)
        else:
            data_to_write[name] = buffers[name].astype(typ[0])

    data_to_write['_'] = x + y + p

    # write data
    data_to_write.tofile(f)
    f.flush()

def read_ncars_dat_int(path):
    events = load_td_data(path, ev_count=-1, ev_start=0)
    return events

class NCARS(Dataset):
    def __init__(self, root, split="train"):
        self.files = []
        for label, sub in enumerate(["background", "cars"]):
            folder = os.path.join(root, split, sub)
            for fname in os.listdir(folder):
                if fname.lower().endswith(".dat"):
                    self.files.append((os.path.join(folder, fname), label))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, label = self.files[idx]
        events = read_ncars_dat_int(path)  # now int64 [x, y, t, p]


        return events, label

'''
def create_one_sample_per_class_grid(dataset_name, start_n,  save_path='class_samples.png'):
    # Load dataset
    if dataset_name == "NMNIST":
        dataset = tonic.datasets.NMNIST(save_to="./data", train=False)
        num_classes = 10
        sensor_size = (34, 34, 2)
        num_ev = 150
        start_n = 0
    elif dataset_name == "NCARS":
        dataset = NCARS("data", split="NCARS/test")
        num_classes = 2  # 0 = background, 1 = car
        sensor_size = (128, 128, 2)
        num_ev = 1000
        start_n = 200
    elif dataset_name == "SNKTH":
        dataset = SNKTH(root="data/SNKTH")
        num_classes = 6  # 0 = background, 1 = car
        sensor_size = (160, 128, 2) #(160, 120, 2)
        num_ev = 1000
        start_n = start_n
    else:
        raise ValueError("Unsupported dataset name")

    # Frame transform
    to_frame = transforms.ToFrame(sensor_size, n_time_bins=1)

    # Dictionary to store one sample per class
    class_samples = {}

    # Loop through dataset to find one sample for each class
    for i in range(len(dataset)):
        events, label = dataset[i]
        if label not in class_samples:
            class_samples[label] = events
            print(label)
        if len(class_samples) == num_classes:
            if len(class_samples[1]) == num_classes and len(class_samples[0]) == num_classes:
                break

    # Plotting
    fig, axs = plt.subplots(1, num_classes, figsize=(num_classes * 2, 2))
    if num_classes == 1:
        axs = [axs]  # make it iterable if only one class

    for i, (label, events) in enumerate(sorted(class_samples.items())):
        frame = to_frame(events[start_n:start_n + num_ev]).squeeze()
        pos, neg = frame[1], frame[0]

        img = np.ones((*pos.shape, 3), dtype=np.uint8) * 255
        img[pos > 0] = [255, 0, 0]  # red
        img[neg > 0] = [0, 0, 255]  # blue

        axs[i].imshow(img)
        axs[i].axis('off')
        # axs[i].set_title(f"Class {label}", fontsize=10)
    del dataset
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")



def create_grid_10_per_class(dataset_name, start_n=0, n_per_class=10, save_path="class_grid.png"):
    # --- pick dataset ---
    if dataset_name == "NMNIST":
        dataset = tonic.datasets.NMNIST(save_to="./data", train=False)
        num_classes = 10
        sensor_size = (34, 34, 2)
        num_ev = 150
    elif dataset_name == "NCARS":
        dataset = NCARS("data", split="NCARS/test")
        num_classes = 2
        sensor_size = (128, 128, 2)
        num_ev = 1000
    elif dataset_name == "SNKTH":
        dataset = SNKTH(root="data/SNKTH")
        num_classes = 6
        sensor_size = (160, 128, 2)   # adjust if yours differs
        num_ev = 1000
    else:
        raise ValueError("Unsupported dataset name")

    to_frame = transforms.ToFrame(sensor_size, n_time_bins=1)

    # --- collect up to n_per_class samples per class ---
    class_buf = {c: [] for c in range(num_classes)}
    for i in range(len(dataset)):
        events, label = dataset[i]
        # keep only up to n_per_class per label
        if len(class_buf[label]) < n_per_class:
            class_buf[label].append(events)
        # stop early when all classes are full
        if all(len(class_buf[c]) >= n_per_class for c in class_buf):
            break

    # sanity-check: some classes might be missing
    missing = [c for c in range(num_classes) if len(class_buf[c]) < n_per_class]
    if missing:
        print(f"[WARN] Not enough samples for classes: {missing}")

    # --- plot grid: rows = classes, cols = n_per_class ---
    fig, axs = plt.subplots(num_classes, n_per_class,
                            figsize=(2*n_per_class, 2*num_classes))
    if num_classes == 1:
        axs = np.expand_dims(axs, 0)
    if n_per_class == 1:
        axs = np.expand_dims(axs, 1)

    for r, cls in enumerate(sorted(class_buf.keys())):
        samples = class_buf[cls]
        for c in range(n_per_class):
            ax = axs[r, c]
            if c < len(samples):
                ev = samples[c]
                # guard for short streams
                end = min(len(ev), start_n + num_ev)
                ev_slice = ev[start_n:end]

                frame = to_frame(ev_slice).squeeze()  # (2,H,W) -> (H,W) after squeeze if n_time_bins=1
                pos, neg = frame[1], frame[0]

                img = np.ones((*pos.shape, 3), dtype=np.uint8) * 255
                img[pos > 0] = [255, 0, 0]   # red for ON
                img[neg > 0] = [0, 0, 255]   # blue for OFF

                ax.imshow(img)
            ax.axis('off')
            if c == 0:
                ax.set_title(f"Class {cls}", fontsize=9, pad=4, loc='left')

    plt.tight_layout(w_pad=0.1, h_pad=0.1)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")'''

import os
import numpy as np
from PIL import Image
import tonic
from tonic import transforms

def save_images_per_class(
    dataset_name: str,
    start_n: int = 0,
    n_per_class: int = 10,          # set up to 100
    out_root: str = "class_samples",
    rotate_180: bool = True
):
    # --- choose dataset & defaults ---
    if dataset_name == "NMNIST":
        dataset = tonic.datasets.NMNIST(save_to="./data", train=False)
        num_classes = 10
        sensor_size = (34, 34, 2)
        num_ev = 150
    elif dataset_name == "NCARS":
        dataset = NCARS("data", split="NCARS/test")
        num_classes = 2
        sensor_size = (128, 128, 2)
        num_ev = 1000
    elif dataset_name == "SNKTH":
        dataset = SNKTH(root="data/SNKTH")
        num_classes = 6
        sensor_size = (160, 128, 2)
        num_ev = 1500
    elif dataset_name == "ASLDVS":
        dataset = ASLDVS(root="data/ASLDVS")
        num_classes = 24
        sensor_size = (240, 180, 2)
        num_ev = 4000
    else:
        raise ValueError("Unsupported dataset name")

    to_frame = transforms.ToFrame(sensor_size, n_time_bins=1)

    # collect indices
    class_indices = {c: [] for c in range(num_classes)}
    for i in range(len(dataset)):
        _, label = dataset[i]
        if len(class_indices[label]) < n_per_class:
            class_indices[label].append(i)
        if all(len(class_indices[c]) >= n_per_class for c in class_indices):
            break

    # save images
    os.makedirs(out_root, exist_ok=True)
    for cls in sorted(class_indices):
        cls_dir = os.path.join(out_root, f"class_{cls}")
        os.makedirs(cls_dir, exist_ok=True)

        for j, idx in enumerate(class_indices[cls]):
            events, _ = dataset[idx]
            end = min(len(events), start_n + num_ev)
            if end <= start_n:
                continue
            ev_slice = events[start_n:end]

            frame = to_frame(ev_slice).squeeze()  # (2,H,W)
            pos, neg = frame[1], frame[0]

            img = np.ones((*pos.shape, 3), dtype=np.uint8) * 255
            img[pos > 0] = [255, 0, 0]
            img[neg > 0] = [0, 0, 255]

            im = Image.fromarray(img)
            if rotate_180:
                im = im.transpose(Image.ROTATE_180)  # 180° rotation

            fn = f"{dataset_name}_c{cls}_{j:03d}.png"
            im.save(os.path.join(cls_dir, fn))

    print(f"Saved rotated images under: {os.path.abspath(out_root)}")


# Generate one image per class for N-MNIST and N-CARS
# create_one_sample_per_class_grid("NMNIST", save_path="nmnist_classes.png")
# create_one_sample_per_class_grid("NCARS", save_path="ncars_classes.png")
X = [0]
for x in X:
    save_images_per_class("ASLDVS", start_n=x, n_per_class = 10, out_root=f"Results")
    # create_one_sample_per_class_grid("SNKTH", start_n=x, save_path=f"Results/snkth_classes{x}.png")