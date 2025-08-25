from src.graph_to_vec_converter import  HVs
from sklearn.metrics       import accuracy_score, classification_report
from sklearn.linear_model  import LogisticRegression
from sklearn.model_selection import train_test_split
from src.graph_generation import NMNISTGraphDataset
from src.loader import ev_loader
from src.graphcnnVSA_Binding_FULL import GraphCNN
from src.codebook import CodeBook
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
from torch.utils.data import Subset
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.pipeline import Pipeline
from datetime import datetime
import os
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.svm  import SVC
from sklearn.model_selection import StratifiedGroupKFold
import time


def main(normalized_feat, num_of_graph_events):
    print("[LOG] - parameter initialization.")
    # GRAPH parameters
    DATA_PATH = "data"
    DATASET = "full"  # full / test      size of dataset loading for training and testing

    NORMALIZE_FEAT = False
    NUM_OF_GRAPH_EVENTS = 100  # None, 10, 50, 100. etc
    R = 4
    D_MAX = 16

    # NOISE parameters
    NOICE_REMOVED = True
    NR_BIN_XY_SIZE = 15
    NR_TIME_BIN_SIZE = 20_000
    NR_MINIMUM_EVENTS = 3

    # GVFA parameters
    LAYERS = 5
    DELTA = 1  # 2
    EQUATION = 11
    DEVICE = torch.device("cpu")


    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = "logs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"clf_svm_{ts}.txt")

    # load event streams
    print("[LOG] - Loading events")
    full_ev_ds = ev_loader(root=DATA_PATH, dataset=DATASET)

    ###################################################################

    indices = np.arange(len(full_ev_ds))
    labels  = [full_ev_ds[i][1] for i in indices]
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=labels
    )

    ds_train = Subset(full_ev_ds, train_idx.tolist())
    ds_test  = Subset(full_ev_ds, test_idx.tolist())

    # ds_train, ds_test = train_test_split(full_ev_ds, test_size=0.2, random_state=10, shuffle=True)   
    ######################################################################
    print("[LOG] - Making class objects.")


    MNISTGraph_model_test_100 = NMNISTGraphDataset(tonic_raw_dataset=ds_test, num_of_graph_events=NUM_OF_GRAPH_EVENTS,
                                                   R=R, Dmax=D_MAX,
                                                   noise_remove=NOICE_REMOVED, normalized_feat=NORMALIZE_FEAT,
                                                   nr_bin_xy_size=NR_BIN_XY_SIZE, nr_minimum_events=NR_MINIMUM_EVENTS,
                                                   nr_time_bin_size=NR_TIME_BIN_SIZE)
    
    HV_Dimensions = [5000, 10000]
    print("Start For loop")
    for HV_DIMENTION in HV_Dimensions:

        gvfa_model = GraphCNN(input_dim=HV_DIMENTION, num_layers=LAYERS, delta=DELTA, graph_pooling_type="sum",
                          neighbor_pooling_type="sum", device=DEVICE, equation=EQUATION).to(DEVICE)
        cb = CodeBook(dim=HV_DIMENTION)
        hvs = HVs(codebook=cb, gvfa_model=gvfa_model)

        X_train_100, X_test_100, X_test_50, X_test_10, Y_train_100, Y_test_100, y_test_50, y_test_10 = [], [], [], [], [], [], [], []
        times=[]
        for i in range(len(ds_test)):
            # print(i)
            t0=time.perf_counter_ns()
            g = MNISTGraph_model_test_100.get(i)
            x, y = hvs.make_hvs(graph=g)
            t1=time.perf_counter_ns()
            times.append((t1-t0)/1e6)

        print("transform time : ", float(np.average(times)))


if __name__ == "__main__":
    main(normalized_feat=False, num_of_graph_events=100)