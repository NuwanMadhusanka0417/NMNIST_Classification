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
    def __init__(self, tonic_raw_dataset,normalized_feat, num_of_graph_events = None, noise_remove = False, nr_bin_xy_size = 5, nr_time_bin_size = 20_000, nr_minimum_events=10):
        super().__init__()
        self.base = tonic_raw_dataset               # produces (events, label)
        self.num_of_graph_events = num_of_graph_events          # Number of events in a graph
        self.noise_remove = noise_remove
        self.normalized_feat = normalized_feat

        # noise removing voxel size parametres
        self.xy_size = nr_bin_xy_size
        self.time_bin_size = nr_time_bin_size
        self.minimum_events =  nr_minimum_events                   # minimum number of events in a voxel

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
        # g = events_to_st_graph(
        #             ev_sized, spatial_thr=3.0, normalized_feat= self.normalized_feat,
        #             time_thr_us=1_000, directed=True)   # Δt ≤ 1 ms
        g = events_to_sota_graph(ev_sized)  # Δt ≤ 1 ms

        g.y = torch.tensor([l], dtype=torch.long)
        return g


full_ev_ds = ev_loader(root="data")

print("Dataset loaded")

normalized_feat = False
num_of_graph_events = 50  # None, 10, 50, 100. etc
MNISTGraph_model   = NMNISTGraphDataset(tonic_raw_dataset=full_ev_ds, num_of_graph_events=num_of_graph_events, noise_remove=False, normalized_feat=normalized_feat)


print("Making graphs....")
full_graph_list = [ MNISTGraph_model.get(i) for i in range(len(full_ev_ds)) ]


print("MaKing graphs completed")
path_to_save = ("data/"+
                str("normalized_graph" if normalized_feat == True else "unnormalized_graph")+ "/sota_graphs_full_E_" +
                (str("all") if num_of_graph_events==None else str(num_of_graph_events))+".pt")

torch.save(full_graph_list, path_to_save)
print("Graphs saved to ->", path_to_save)