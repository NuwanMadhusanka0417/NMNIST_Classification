import torch
from src.graphcnnVSA_Binding_FULL import GraphCNN
from src.codebook import CodeBook

def make_hvs(graphs, normalized_feat, num_of_graph_events):
    # 1. initialize parameters
    dimension = 5000
    layer = 5
    delta = 1 # 2
    equation = 11
    device = torch.device("cpu")

    # 2. load classes
    gvfa_model = GraphCNN(input_dim=dimension, num_layers=layer, delta=delta, graph_pooling_type="sum",
                          neighbor_pooling_type="sum", device=device, equation=equation).to(device)
    cb = CodeBook(dim=dimension)

    print("HV dimention ", dimension)
    # 3. Load graph data

    # 4. Make hyper vectros
    X = []
    Y = []

    for g in graphs:
        all_nodes = []
        for node in g.x:
            x_, y_, t_, p_ = node.tolist()
            x_hv = cb.HV_X[int(x_)]
            y_hv = cb.HV_Y[int(y_)]
            t_hv = cb.HV_T[(int(t_) // cb.t_step) * cb.t_step]
            if int(p_) == -1:
                p_ = 0
            else:
                p_ = 1
            p_hv = cb.HV_P[int(p_)]

            all_nodes.append(cb.bundle([x_hv, y_hv, t_hv, p_hv]))

        graph_hv = gvfa_model(torch.stack(all_nodes), g.edge_index)

        X.append(graph_hv)
        Y.append(g.y.item())

    ## Save Hyper Vectors
    # path_to_save = ("data/" +
    #                 str("normalized_graph" if normalized_feat == True else "unnormalized_graph") + "/hyper_vectors_E_" +
    #                 (str("all") if num_of_graph_events == None else str(num_of_graph_events))+".pt")
    # torch.save({
    #     'hvs': X,
    #     'labels': Y
    # }, path_to_save)

    return X, Y