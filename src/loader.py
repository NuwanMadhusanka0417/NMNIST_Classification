import tonic
import torch
import tonic.datasets.nmnist
import tonic.transforms as transforms
from torch.utils.data import ConcatDataset
from src.ncars_data_loader import NCARS

def ev_loader(root:str = 'data', dataset = "full"):

    if dataset == "full":
        train_ds = NCARS(root, split="NCARS/train")
        test_ds = NCARS(root, split="NCARS/test")
        print("LOG: load full dataset")
        return  train_ds, test_ds #ConcatDataset((train_ds, test_ds))

    else:
        test_ds = NCARS(root, split="NCARS/test")
        return test_ds


def graph_loader(normalized_feat = False, num_of_graph_events = None):
    DATASET = "full"
    NORMALIZE_FEAT = normalized_feat
    NUM_OF_GRAPH_EVENTS = num_of_graph_events  # None, 10, 50, 100. etc
    R = 4
    D_MAX = 16

    NOICE_REMOVED = True
    NR_BIN_XY_SIZE = 15
    NR_TIME_BIN_SIZE = 20_000
    NR_MINIMUM_EVENTS = 3

    path_to_save = ("data/" +
                    str("normalized_graph" if NORMALIZE_FEAT == True else "unnormalized_graph") + "/R_mthd_graphs_" + DATASET +
                    "_R" + str(R) + "_Dmax" + str(D_MAX) +
                    "_NR_" + str("t" if NOICE_REMOVED == True else "f") + "_NRxy" + str(NR_BIN_XY_SIZE) + "_NRt" + str(
                NR_TIME_BIN_SIZE) + "_NRmine" + str(NR_MINIMUM_EVENTS) +
                    "_E_" + (str("all") if NUM_OF_GRAPH_EVENTS == None else str(NUM_OF_GRAPH_EVENTS)) + ".pt")

    print("LOG-[Training & Testing]: NUM_OF_GRAPH_EVENTS:", NUM_OF_GRAPH_EVENTS, " | DATASET:", DATASET,
          " | NORMALIZE_FEAT:", NORMALIZE_FEAT,
          " | R:", R, " | D_MAX: ", D_MAX, " | NOICE_REMOVED: ", NOICE_REMOVED,
          " | NR_BIN_XY_SIZE: ", NR_BIN_XY_SIZE, " | NR_TIME_BIN_SIZE: ", NR_TIME_BIN_SIZE, " | NR_MINIMUM_EVENTS: ",
          NR_MINIMUM_EVENTS)
    print("LOG - loaded graph: ", path_to_save)
    return torch.load(path_to_save)
