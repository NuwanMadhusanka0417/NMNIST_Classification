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



    # load event streams
    print("[LOG] - Loading events")
    full_ev_ds = ev_loader(root=DATA_PATH, dataset=DATASET)
    ds_train, ds_test = train_test_split(full_ev_ds, test_size=0.2, random_state=42, shuffle=True)   

    print("[LOG] - Making class objects.")

    MNISTGraph_model_train_100 = NMNISTGraphDataset(tonic_raw_dataset=ds_train, num_of_graph_events=NUM_OF_GRAPH_EVENTS,
                                                    R=R, Dmax=D_MAX,
                                                    noise_remove=NOICE_REMOVED, normalized_feat=NORMALIZE_FEAT,
                                                    nr_bin_xy_size=NR_BIN_XY_SIZE, nr_minimum_events=NR_MINIMUM_EVENTS,
                                                    nr_time_bin_size=NR_TIME_BIN_SIZE)

    MNISTGraph_model_test_100 = NMNISTGraphDataset(tonic_raw_dataset=ds_test, num_of_graph_events=NUM_OF_GRAPH_EVENTS,
                                                   R=R, Dmax=D_MAX,
                                                   noise_remove=NOICE_REMOVED, normalized_feat=NORMALIZE_FEAT,
                                                   nr_bin_xy_size=NR_BIN_XY_SIZE, nr_minimum_events=NR_MINIMUM_EVENTS,
                                                   nr_time_bin_size=NR_TIME_BIN_SIZE)

    MNISTGraph_model_test_50 = NMNISTGraphDataset(tonic_raw_dataset=ds_test, num_of_graph_events=50,
                                                  R=R, Dmax=D_MAX,
                                                  noise_remove=NOICE_REMOVED, normalized_feat=NORMALIZE_FEAT,
                                                  nr_bin_xy_size=NR_BIN_XY_SIZE, nr_minimum_events=NR_MINIMUM_EVENTS,
                                                  nr_time_bin_size=NR_TIME_BIN_SIZE)

    MNISTGraph_model_test_10 = NMNISTGraphDataset(tonic_raw_dataset=ds_test, num_of_graph_events=10,
                                                  R=R, Dmax=D_MAX,
                                                  noise_remove=NOICE_REMOVED, normalized_feat=NORMALIZE_FEAT,
                                                  nr_bin_xy_size=NR_BIN_XY_SIZE, nr_minimum_events=NR_MINIMUM_EVENTS,
                                                  nr_time_bin_size=NR_TIME_BIN_SIZE)
    
    HV_Dimensions = [1000, 3000, 5000, 10000, 15000]
    print("Start For loop")
    for HV_DIMENTION in HV_Dimensions:

        gvfa_model = GraphCNN(input_dim=HV_DIMENTION, num_layers=LAYERS, delta=DELTA, graph_pooling_type="sum",
                          neighbor_pooling_type="sum", device=DEVICE, equation=EQUATION).to(DEVICE)
        cb = CodeBook(dim=HV_DIMENTION)
        hvs = HVs(codebook=cb, gvfa_model=gvfa_model)

        X_train_100, X_test_100, X_test_50, X_test_10, Y_train_100, Y_test_100, y_test_50_10 = [], [], [], [], [], [], []

        for i in range(len(ds_train)):
            # print(i)
            g = MNISTGraph_model_train_100.get(i)
            x, y = hvs.make_hvs(graph=g)
            X_train_100.append(x)
            Y_train_100.append(y)
        for i in range(len(ds_test)):
            # print(i)
            g = MNISTGraph_model_test_100.get(i)
            x, y = hvs.make_hvs(graph=g)
            X_test_100.append(x)
            Y_test_100.append(y)

        for i in range(len(ds_test)):
            # print(i)
            g_50 = MNISTGraph_model_test_50.get(i)
            g_10 = MNISTGraph_model_test_10.get(i)

            # print(g)

            x_50, y = hvs.make_hvs(graph=g_50)
            x_10, _ = hvs.make_hvs(graph=g_10)

            X_test_50.append(x_50)
            X_test_10.append(x_10)
            y_test_50_10.append(y)
        print("Start Classification")
        clf = LogisticRegression(
            solver='saga',       # handles high-dimensional sparse data
            penalty='l2',        # ridge regularization
            n_jobs=-1,           # parallelize over cores
            random_state=42
        )

        # 2. Pipeline (with scaling for sparse/hypervector inputs)
        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("lr",    clf)
        ])

        # 3. Expanded hyperparameter grid (now including max_iter)
        param_grid = {
            "lr__C":            [ 0.1, 1],  # 0.1
            "lr__tol":          [ 1e-3, 1e-2],
            "lr__class_weight": [None],
            "lr__max_iter":     [1000, 2000]
        }

        # 4. GridSearchCV setup
        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            cv=5,                 # 5-fold CV
            scoring="accuracy",
            n_jobs=-1,
            verbose=1
        )

        grid.fit(X_train_100, Y_train_100)
        print(grid.best_params_)
        print(grid.best_score_)
        print(grid.param_grid)
        

        print("----100------")
        print(f"Train accuracy: {accuracy_score(Y_train_100, grid.predict(X_train_100)) * 100:.2f}%")
        print(f"Test  accuracy: {accuracy_score(Y_test_100, grid.predict(X_test_100)) * 100:.2f}%")

        print("----50------")
        print(f"Test  accuracy: {accuracy_score(y_test_50_10, grid.predict(X_test_50)) * 100:.2f}%")

        print("----10------")
        print(f"Test  accuracy: {accuracy_score(y_test_50_10, grid.predict(X_test_10)) * 100:.2f}%")

        print("[LOG]- NUM_OF_GRAPH_EVENTS:", NUM_OF_GRAPH_EVENTS, " | DATASET:", DATASET,
              " | NORMALIZE_FEAT:", NORMALIZE_FEAT,
              " | R:", R, " | D_MAX: ", D_MAX, " | NOICE_REMOVED: ", NOICE_REMOVED,
              " | NR_BIN_XY_SIZE: ", NR_BIN_XY_SIZE, " | NR_TIME_BIN_SIZE: ", NR_TIME_BIN_SIZE,
              " | NR_MINIMUM_EVENTS: ",
              NR_MINIMUM_EVENTS, " | HV_DIMENTION: ", HV_DIMENTION, " | LAYERS: ", LAYERS, " | DELTA: ", DELTA,
              " | EQUATION: ", EQUATION, )
        
        del gvfa_model
        del cb
        del hvs
        del X_train_100
        del X_test_100
        del X_test_50
        del X_test_10
        del Y_train_100
        del Y_test_100
        del y_test_50_10
        del grid


if __name__ == "__main__":
    main(normalized_feat=False, num_of_graph_events=100)