from src.graph_to_vec_converter import  HVs
from sklearn.metrics       import accuracy_score, classification_report
from sklearn.linear_model  import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from src.graph_generation import NMNISTGraphDataset
from src.loader import ev_loader
from src.graphcnnVSA_Binding_FULL import GraphCNN
from src.codebook import CodeBook
import torch
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing   import StandardScaler
from sklearn.svm             import SVC
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import accuracy_score
from torch.utils.data import ConcatDataset
import numpy as np

print("[LOG] - parameter initialization.")
# GRAPH parameters
DATA_NAME = "NCARS" # NCARS, NMNIST
DATA_PATH = "data"
DATASET = "full"  # full / test      size of dataset loading for training and testing

if DATA_NAME == "NCARS":
    X_MAX = 360
    Y_MAX = 360
    T_MAX = 100_000_000
    T_STEP = 10_000

NORMALIZE_FEAT = False
NUM_OF_GRAPH_EVENTS = 100  # None, 10, 50, 100. etc
R = 7
D_MAX = 30

# NOISE parameters
NOICE_REMOVED = True
NR_BIN_XY_SIZE = 4
NR_TIME_BIN_SIZE = 20_000
NR_MINIMUM_EVENTS = 3

# GVFA parameters
HV_DIMENTION = 1000
LAYERS = 5
DELTA = 1  # 2
EQUATION = 11
DEVICE = torch.device("cpu")

# load event streams
print("[LOG] - Loading events")
# full_ev_ds = ev_loader(root=DATA_PATH, dataset=DATASET)
ds = ev_loader(root=DATA_PATH, dataset=DATASET)

train_ds, test_ds = train_test_split(ds, test_size=0.2, random_state=10, shuffle=True)   
print("[LOG] - Making class objects.")
MNISTGraph_model_train = NMNISTGraphDataset(tonic_raw_dataset=ds, num_of_graph_events=NUM_OF_GRAPH_EVENTS,
                                      R=R, Dmax=D_MAX,
                                      noise_remove=NOICE_REMOVED, normalized_feat=NORMALIZE_FEAT,
                                      nr_bin_xy_size=NR_BIN_XY_SIZE, nr_minimum_events=NR_MINIMUM_EVENTS,
                                      nr_time_bin_size=NR_TIME_BIN_SIZE)



MNISTGraph_model_test_50 = NMNISTGraphDataset(tonic_raw_dataset=test_ds, num_of_graph_events=50,
                                            R=R, Dmax=D_MAX,
                                            noise_remove=NOICE_REMOVED, normalized_feat=NORMALIZE_FEAT,
                                            nr_bin_xy_size=NR_BIN_XY_SIZE, nr_minimum_events=NR_MINIMUM_EVENTS,
                                            nr_time_bin_size=NR_TIME_BIN_SIZE)

MNISTGraph_model_test_10 = NMNISTGraphDataset(tonic_raw_dataset=test_ds, num_of_graph_events=10,
                                            R=R, Dmax=D_MAX,
                                            noise_remove=NOICE_REMOVED, normalized_feat=NORMALIZE_FEAT,
                                            nr_bin_xy_size=NR_BIN_XY_SIZE, nr_minimum_events=NR_MINIMUM_EVENTS,
                                            nr_time_bin_size=NR_TIME_BIN_SIZE)

items = [1000]
for item in items:
    HV_DIMENTION = item
    gvfa_model = GraphCNN(input_dim=HV_DIMENTION, num_layers=LAYERS, delta=DELTA, graph_pooling_type="sum",
                          neighbor_pooling_type="sum", device=DEVICE, equation=EQUATION).to(DEVICE)
    cb = CodeBook(dim=HV_DIMENTION, x_max=X_MAX, y_max=Y_MAX, t_max=T_MAX, t_step=T_STEP)
    hvs = HVs(codebook=cb, gvfa_model=gvfa_model)


    X_,X_test_50_,X_test_10_, Y_, y_test_50_10 = [],[],[],[], []
    print("[LOG] - Loading graph and converting to HVs.")
    for i in range(len(ds)):
        # print(i)
        g = MNISTGraph_model_train.get(i)
        x, y = hvs.make_hvs(graph=g)
        X_.append(x)
        Y_.append(y)


    X_train_, X_test_, y_train, y_test = train_test_split(X_, Y_, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_)
    X_test = scaler.transform(X_test_)

    np.savez_compressed("data/NMNIST_hv/train_ncars_100.npz", X=X_train, y=y_train)
    np.savez_compressed("data/NMNIST_hv/test_ncars_100.npz",  X=X_test,  y=y_test)

    print("HVs saved")
    for i in range(len(test_ds)):
        # print(i)
        g_50 = MNISTGraph_model_test_50.get(i)
        g_10 = MNISTGraph_model_test_10.get(i)

        # print(g)

        x_50, y = hvs.make_hvs(graph=g_50)
        x_10, _ = hvs.make_hvs(graph=g_10)


        X_test_50_.append(x_50)
        X_test_10_.append(x_10)
        y_test_50_10.append(y)


    X_test_50 = scaler.fit_transform(X_test_50_)
    X_test_10 = scaler.fit_transform(X_test_10_)


    del cb
    del hvs
    del gvfa_model
    # del full_ev_ds

    # el = [1000, 5000]

    print("[LOG] - Classification.")

    # clf = SVC(kernel="rbf", C=0.1, gamma=0.9,degree=6)
    CS = [3,4,5, 6]
    for c in CS:
        grid = SVC(kernel="rbf", C=c, gamma='scale', class_weight="balanced")

        grid.fit(X_train, y_train)

        tr_acc = f"{accuracy_score(y_train, grid.predict(X_train)) * 100:.2f}%"
        ts_100 = f"{accuracy_score(y_test, grid.predict(X_test)) * 100:.2f}%"
        ts_50 =  f"{accuracy_score(y_test_50_10, grid.predict(X_test_50)) * 100:.2f}%"
        ts_10 = f"{accuracy_score(y_test_50_10, grid.predict(X_test_10)) * 100:.2f}%"

        print(f"{HV_DIMENTION}, {c}, {tr_acc}, {ts_100}, {ts_50}, {ts_10}")

        # print("----100------")
        # print(f"Train accuracy: {accuracy_score(y_train, grid.predict(X_train)) * 100:.2f}%")
        # print(f"Test  accuracy: {accuracy_score(y_test, grid.predict(X_test)) * 100:.2f}%")

        # print("----50------")
        # print(f"Test  accuracy: {accuracy_score(y_test_50_10, grid.predict(X_test_50)) * 100:.2f}%")

        # print("----10------")
        # print(f"Test  accuracy: {accuracy_score(y_test_50_10, grid.predict(X_test_10)) * 100:.2f}%")

        # print("[LOG]- NUM_OF_GRAPH_EVENTS:", NUM_OF_GRAPH_EVENTS, " | DATASET:", DATASET,
        #     " | NORMALIZE_FEAT:", NORMALIZE_FEAT,
        #     " | R:", R, " | D_MAX: ", D_MAX, " | NOICE_REMOVED: ", NOICE_REMOVED,
        #     " | NR_BIN_XY_SIZE: ", NR_BIN_XY_SIZE, " | NR_TIME_BIN_SIZE: ", NR_TIME_BIN_SIZE, " | NR_MINIMUM_EVENTS: ",
        #     NR_MINIMUM_EVENTS, " | HV_DIMENTION: ", HV_DIMENTION," | LAYERS: ", LAYERS," | DELTA: ", DELTA," | EQUATION: ", EQUATION,)


        # print(f)

        # del clf
