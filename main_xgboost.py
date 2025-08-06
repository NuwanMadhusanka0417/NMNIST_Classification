
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
from xgboost import XGBClassifier

def main():


    print("[LOG] - parameter initialization.")
    # GRAPH parameters
    DATA_NAME = "SNKTH"  # NCARS, NMNIST
    DATA_PATH = "data"
    DATASET = "full"  # full / test      size of dataset loading for training and testing
    NORMALIZE_FEAT = False
    NUM_OF_GRAPH_EVENTS = 100  # None, 10, 50, 100. etc

    if DATA_NAME == "SNKTH":
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
    ds_train, ds_test = train_test_split(ds, test_size=0.2, random_state=42, shuffle=True)

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

    HV_Dimensions = [500, 1000, 5000, 7000]
    for item in HV_Dimensions:
        HV_DIMENTION = item
        gvfa_model = GraphCNN(input_dim=HV_DIMENTION, num_layers=LAYERS, delta=DELTA, graph_pooling_type="sum",
                              neighbor_pooling_type="sum", device=DEVICE, equation=EQUATION).to(DEVICE)
        cb = CodeBook(dim=HV_DIMENTION, x_max=X_MAX, y_max=Y_MAX, t_max=T_MAX, t_step=T_STEP)
        hvs = HVs(codebook=cb, gvfa_model=gvfa_model)

        X_train_100, X_test_100, X_test_50, X_test_10, Y_train_100, Y_test_100, y_test_50_10 = [], [], [], [], [], [], []
        print("[LOG] - Loading graph and converting to HVs.")
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

        # scaler = StandardScaler()
        # X_train = scaler.fit_transform(X_train_)
        # X_test = scaler.fit_transform(X_test_)

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

        # X_test_50 = scaler.fit_transform(X_test_50_)
        # X_test_10 = scaler.fit_transform(X_test_10_)

        del cb
        del hvs
        del gvfa_model
        # del full_ev_ds

        print("[LOG] - Classification.")

        # clf = SVC(kernel="rbf", C=0.1, gamma=0.9,degree=6)
        pipe_xgb = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("xgb", XGBClassifier(
                num_class= 6,
                objective="multi:softmax",  # or "multi:softprob" for probability output
                use_label_encoder=False,
                eval_metric="mlogloss"
            ))
        ])

        # Define parameter grid for GridSearchCV
        param_grid = {
            "xgb__n_estimators": [50, 100],
            "xgb__max_depth": [3, 6, 10],
            "xgb__learning_rate": [0.01, 0.1, 0.2],
            "xgb__subsample": [0.8, 1.0],
            "xgb__colsample_bytree": [0.8, 1.0]
        }

        # Define grid search
        grid = GridSearchCV(
            pipe_xgb,
            param_grid=param_grid,
            cv=5,
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

        # del clf


if __name__ == "__main__":
    main()