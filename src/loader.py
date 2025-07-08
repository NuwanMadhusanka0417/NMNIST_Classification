import tonic
import torch
import tonic.transforms as transforms
from torch.utils.data import ConcatDataset


def ev_loader(root:str = 'data'):
    train_ds = tonic.datasets.NMNIST(root, train=True)  # <- uses your .bin files
    test_ds = tonic.datasets.NMNIST(root, train=False)
    print("test Datset size: ", len(test_ds))
    return test_ds #ConcatDataset((train_ds, test_ds))


def graph_loader(normalized_feat = False, num_of_graph_events = None):
    path_to__load = ("data/" +
                         str("normalized_graph" if normalized_feat == True else "unnormalized_graph") + "/R_mthd_graphs_test_E_" +
                         (str("all") if num_of_graph_events == None else str(num_of_graph_events))+".pt")

    return torch.load(path_to__load)
