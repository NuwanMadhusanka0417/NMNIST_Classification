from torch_geometric.data import Data, Dataset
from scipy.spatial import cKDTree
from typing import Tuple
import numpy as np
import torch
import tonic
from src.noise_remover import purge_sparse_voxels
from src.events_to_graph_converter import events_to_st_graph, events_to_sota_graph
from src.loader import ev_loader

class NMNISTGraphDataset(Dataset):
    def __init__(self, tonic_raw_dataset,normalized_feat, R, Dmax, num_of_graph_events = None, noise_remove = False, nr_bin_xy_size = 6, nr_time_bin_size = 20_000, nr_minimum_events=10):
        super().__init__()
        self.base = tonic_raw_dataset               # produces (events, label)
        self.num_of_graph_events = num_of_graph_events          # Number of events in a graph
        self.noise_remove = noise_remove
        self.normalized_feat = normalized_feat

        # noise removing voxel size parametres
        self.xy_size = nr_bin_xy_size
        self.time_bin_size = nr_time_bin_size
        self.minimum_events =  nr_minimum_events                   # minimum number of events in a voxel


        self.R = R
        self.Dmax = Dmax
    def filter_events(self, ev, size):
        times = ev['t']  # shape (N,)
        mask_ = times > 120_000  # shape (N,) of dtype torch.bool
        ev_after_ = ev[mask_]

        # 2) take at most the first 100 of them
        return ev_after_[:size]

    def len(self):
        return len(self.base)

    def get(self, idx):
        event, l = self.base[idx]
        print(idx)

        ## NOISE REMOVE USING VOXELS - divide events into voxel and remove voxel with,  number of events < min_events
        if self.noise_remove:
            events_nr = purge_sparse_voxels(
                        event, sensor_size=(34,34),
                        xy_bin_size=self.xy_size, time_bin_us=self.time_bin_size, min_events=self.minimum_events)
        else:
            events_nr = event

        ## EVENTS FILTER - (not voxel filter)
        if self.num_of_graph_events is not None:
            ev_sized = self.filter_events(events_nr, self.num_of_graph_events)
        else:
            ev_sized = events_nr

        ## MAKE GRAPH
        g = events_to_st_graph(
                    ev_sized, R=self.R,Dmax=self.Dmax, normalized_feat= self.normalized_feat,
                    directed=True)   # Δt ≤ 1 ms

        g.y = torch.tensor([l], dtype=torch.long)
        return g


full_ev_ds = ev_loader(root="data")

print("Dataset loaded")

NORMALIZE_FEAT = False
NUM_OF_GRAPH_EVENTS = 100  # None, 10, 50, 100. etc
R = 4
D_MAX = 16
NOICE_REMOVED = False

MNISTGraph_model   = NMNISTGraphDataset(tonic_raw_dataset=full_ev_ds, num_of_graph_events=NUM_OF_GRAPH_EVENTS, R=R, Dmax=D_MAX, noise_remove=NOICE_REMOVED, normalized_feat=NORMALIZE_FEAT)


print("Making graphs....")
full_graph_list = [ MNISTGraph_model.get(i) for i in range(len(full_ev_ds)) ]


print("MaKing graphs completed")
path_to_save = ("data/"+
                str("normalized_graph" if NORMALIZE_FEAT == True else "unnormalized_graph")+ "/R_mthd_graphs_test_s4_E_" +
                (str("all") if NUM_OF_GRAPH_EVENTS==None else str(NUM_OF_GRAPH_EVENTS))+".pt")

torch.save(full_graph_list, path_to_save)
print("Graphs saved to ->", path_to_save)