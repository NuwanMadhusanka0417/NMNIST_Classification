import torch
from src.graphcnnVSA_Binding_FULL import GraphCNN
from src.codebook import CodeBook
class HVs():
    def __init__(self, gvfa_model, codebook):
        self.gvfa = gvfa_model
        self.cb = codebook

    def make_hvs(self, graph):

        all_nodes = []
        for node in graph.x:
            x_, y_, t_, p_ = node.tolist()
            x_hv = self.cb.HV_X[int(x_)]
            y_hv = self.cb.HV_Y[int(y_)]
            t_hv = self.cb.HV_T[(int(t_) // self.cb.t_step) * self.cb.t_step]
            if int(p_) == -1:
                p_ = 0
            else:
                p_ = 1
            p_hv = self.cb.HV_P[int(p_)]

            all_nodes.append(self.cb.bundle([x_hv, y_hv, t_hv, p_hv]))

        graph_hv = self.gvfa(torch.stack(all_nodes), graph.edge_index)

        return graph_hv, graph.y.item()