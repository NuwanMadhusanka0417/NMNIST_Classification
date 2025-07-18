import os
import struct
import numpy as np
from torch.utils.data import Dataset, DataLoader
from src.dat_events_tools import load_td_data


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


# Usage example
if __name__ == "__main__":
    root = "../data"   # <-- change this to your data folder
    train_ds = NCARS(root, split="NCARS/train")
    test_ds  = NCARS(root, split="NCARS/test")

    e, l = train_ds[0]
    print(e, l)

    print(" number of samples = ", len(train_ds))
    print(" number of samples = ", len(test_ds))

    print(len(train_ds[0]))
    print(len(train_ds[0]))
    print(len(train_ds[0][0]))
    print(len(train_ds[0][0][0]))
    print(train_ds[0][0][0])
    print(train_ds[0][0][1])
    print(train_ds[0][0][-1])





