
from src.graph_to_vec_converter import HVs
from sklearn.linear_model import LogisticRegression
from src.graph_generation import NMNISTGraphDataset
from src.loader import ev_loader
from src.graphcnnVSA_Binding_FULL import GraphCNN
from src.codebook import CodeBook
import torch
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import numpy as np




print("[LOG] - parameter initialization.")
# GRAPH parameters
DATA_NAME = "ASLDVS"  # NCARS, NMNIST
DATA_PATH = "data"# "/scratch/mi23/nk8155/datasets" #"data"
DATASET = "full"  # full / test      size of dataset loading for training and testing
NORMALIZE_FEAT = False
NUM_OF_GRAPH_EVENTS = 9000  # None, 10, 50, 100. etc

if DATA_NAME == "ASLDVS":
    X_MAX = 360
    Y_MAX = 360
    T_MAX = 1_000_000
    T_STEP = 10_0

    R = 2
    D_MAX = 4

    # NOISE parameters
    NOICE_REMOVED = False
    NR_BIN_XY_SIZE = 5
    NR_TIME_BIN_SIZE = 20_00
    NR_MINIMUM_EVENTS = 2

# GVFA parameters
# HV_DIMENTION = 5000
LAYERS = 5
DELTA = 1  # 2
EQUATION = 11
DEVICE = torch.device("cpu")

# load event streams
print("[LOG] - Loading events")
# full_ev_ds = ev_loader(root=DATA_PATH, dataset=DATASET)
ds = ev_loader(root=DATA_PATH, dataset=DATASET, data_name=DATA_NAME)


print(ds)
print(ds[100][0])
print(ds[100][0][-1][0])
min_evs = len(ds[1][0])
max_evs = len(ds[1][0])

min_time = ds[1][0][-1][0]
max_time = ds[1][0][-1][0]
for i in range(len(ds)):
    # print(len(ds[i][0]))
    if min_evs > len(ds[i][0]):
        min_evs = len(ds[i][0])
    if max_evs < len(ds[i][0]):
        max_evs = len(ds[i][0])

    ts_range = ds[i][0][-1][0] - ds[i][0][0][0]

    if min_time > ts_range:
        min_time = ts_range
    if max_time < ts_range:
        max_time = ts_range

    

print("Number of events in sample; min:", min_evs)
print("Number of events in sample; max:", max_evs)

print("Sample time range; min:", min_time)
print("Sample time range; max:", max_time)