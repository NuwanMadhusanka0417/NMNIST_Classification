
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


def main():


    print("[LOG] - parameter initialization.")
    # GRAPH parameters
    DATA_NAME = "ASLDVS"  # NCARS, NMNIST
    DATA_PATH = "data"
    DATASET = "full"  # full / test      size of dataset loading for training and testing
    NORMALIZE_FEAT = False
    NUM_OF_GRAPH_EVENTS = 100  # None, 10, 50, 100. etc

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
    # print(len(ds))
    # ls = []
    # for i in range (100000):
    #     _, l = ds[i]
    #     ls.append(l)

    # print(ls)

    labels = [ds[i][1] for i in range(len(ds))]

    # unique labels (sorted)
    unique_labels = sorted(set(labels))
    print("Unique labels:", unique_labels)
    print("#classes:", len(unique_labels))


main()