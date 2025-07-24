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

def main():
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
    train_ds, test_ds = ev_loader(root=DATA_PATH, dataset=DATASET)

    print("[LOG] - Making class objects.")
    MNISTGraph_model_train = NMNISTGraphDataset(tonic_raw_dataset=train_ds, num_of_graph_events=NUM_OF_GRAPH_EVENTS,
                                          R=R, Dmax=D_MAX,
                                          noise_remove=NOICE_REMOVED, normalized_feat=NORMALIZE_FEAT,
                                          nr_bin_xy_size=NR_BIN_XY_SIZE, nr_minimum_events=NR_MINIMUM_EVENTS,
                                          nr_time_bin_size=NR_TIME_BIN_SIZE)

    MNISTGraph_model_test = NMNISTGraphDataset(tonic_raw_dataset=test_ds, num_of_graph_events=NUM_OF_GRAPH_EVENTS,
                                                R=R, Dmax=D_MAX,
                                                noise_remove=NOICE_REMOVED, normalized_feat=NORMALIZE_FEAT,
                                                nr_bin_xy_size=NR_BIN_XY_SIZE, nr_minimum_events=NR_MINIMUM_EVENTS,
                                                nr_time_bin_size=NR_TIME_BIN_SIZE)

    gvfa_model = GraphCNN(input_dim=HV_DIMENTION, num_layers=LAYERS, delta=DELTA, graph_pooling_type="sum",
                          neighbor_pooling_type="sum", device=DEVICE, equation=EQUATION).to(DEVICE)
    cb = CodeBook(dim=HV_DIMENTION, x_max=X_MAX, y_max=Y_MAX, t_max=T_MAX, t_step=T_STEP)
    hvs = HVs(codebook=cb, gvfa_model=gvfa_model)


    X_train, X_test, y_train, y_test = [],[],[],[]
    print("[LOG] - Loading graph and converting to HVs.")
    for i in range(len(train_ds)):
        print(i)
        g = MNISTGraph_model_train.get(i)
        # print(g)
        x, y = hvs.make_hvs(graph=g)
        X_train.append(x)
        y_train.append(y)

    for i in range(len(test_ds)):
        print(i)
        g = MNISTGraph_model_test.get(i)
        # print(g)
        x, y = hvs.make_hvs(graph=g)
        X_test.append(x)
        y_test.append(y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    del cb
    del hvs
    del gvfa_model
    # del full_ev_ds

    el = [1000]
    for elm in el:
        print(elm)
        print("[LOG] - Classification.")

        pipe_rbf = Pipeline([
            ("sc", StandardScaler(with_mean=False)),
            ("svc", SVC(kernel="rbf", ga))
        ])

        param_grid = {
            "svc__C": [0.1, 1, 3, 10],
            "svc__gamma": ["scale", 1e-3, 1e-2, 1e-1],
            "svc__class_weight": [None, "balanced"]
        }

        grid = GridSearchCV(pipe_rbf,
                            param_grid=param_grid,
                            cv=5,
                            n_jobs=-1,
                            verbose=1)
        grid.fit(X_train, y_train)

        print("Best params :", grid.best_params_)

        print(f"Train accuracy: {accuracy_score(y_train, grid.predict(X_train)) * 100:.2f}%")
        print(f"Test  accuracy: {accuracy_score(y_test, grid.predict(X_test)) * 100:.2f}%")

        print("[LOG]- NUM_OF_GRAPH_EVENTS:", NUM_OF_GRAPH_EVENTS, " | DATASET:", DATASET,
              " | NORMALIZE_FEAT:", NORMALIZE_FEAT,
              " | R:", R, " | D_MAX: ", D_MAX, " | NOICE_REMOVED: ", NOICE_REMOVED,
              " | NR_BIN_XY_SIZE: ", NR_BIN_XY_SIZE, " | NR_TIME_BIN_SIZE: ", NR_TIME_BIN_SIZE, " | NR_MINIMUM_EVENTS: ",
              NR_MINIMUM_EVENTS, " | HV_DIMENTION: ", HV_DIMENTION," | LAYERS: ", LAYERS," | DELTA: ", DELTA," | EQUATION: ", EQUATION,)

        # del clf

if __name__ == "__main__":
    main()