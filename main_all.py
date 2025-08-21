
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
    DATA_NAME = "SNKTH"  # NCARS, NMNIST
    DATA_PATH = "data"
    DATASET = "full"  # full / test      size of dataset loading for training and testing
    NORMALIZE_FEAT = False
    NUM_OF_GRAPH_EVENTS = None  # None, 10, 50, 100. etc

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
    #ds_train, ds_test = train_test_split(ds, test_size=0.2, random_state=42, shuffle=True)

    print("[LOG] - Making class objects.")
    MNISTGraph_model_train = NMNISTGraphDataset(tonic_raw_dataset=ds, num_of_graph_events=NUM_OF_GRAPH_EVENTS,
                                                    R=R, Dmax=D_MAX,
                                                    noise_remove=NOICE_REMOVED, normalized_feat=NORMALIZE_FEAT,
                                                    nr_bin_xy_size=NR_BIN_XY_SIZE, nr_minimum_events=NR_MINIMUM_EVENTS,
                                                    nr_time_bin_size=NR_TIME_BIN_SIZE)


    HV_Dimensions = [1000, 5000, 7000]
    for item in HV_Dimensions:
        HV_DIMENTION = item
        gvfa_model = GraphCNN(input_dim=HV_DIMENTION, num_layers=LAYERS, delta=DELTA, graph_pooling_type="sum",
                              neighbor_pooling_type="sum", device=DEVICE, equation=EQUATION).to(DEVICE)
        cb = CodeBook(dim=HV_DIMENTION, x_max=X_MAX, y_max=Y_MAX, t_max=T_MAX, t_step=T_STEP)
        hvs = HVs(codebook=cb, gvfa_model=gvfa_model)

        X, Y = [], []
        print("[LOG] - Loading graph and converting to HVs.")
        for i in range(len(ds)):
            # print(i)
            g = MNISTGraph_model_train.get(i)
            x, y = hvs.make_hvs(graph=g)
            X.append(x)
            Y.append(y)

        X_train_, X_test_, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_)
        X_test = scaler.transform(X_test_)

        print("Start Classification")
        CS = [1, 3 , 4, 5]
        iterations = [120, 200, 500, 800]
        for c in CS:
            '''for iters in iterations:
                print("C = ", c)
                clf = LogisticRegression(
                    C=c,
                    solver='saga',       # handles high-dimensional sparse data
                    penalty='l2',        # ridge regularization
                    n_jobs=-1,           # parallelize over cores
                    max_iter=iters,
                    random_state=42,
                    class_weight='balanced'
                )

                clf.fit(X_train, y_train)

                tr_acc = f"{accuracy_score(y_train, clf.predict(X_train)) * 100:.2f}%"
                ts_acc = f"{accuracy_score(y_test, clf.predict(X_test)) * 100:.2f}%"

                print(f"SNKTH-lgst {HV_DIMENTION}, {c}, {tr_acc}, {ts_acc}")

                del clf'''

            grid = SVC(kernel="rbf", C=c, gamma='scale', class_weight="balanced")

            grid.fit(X_train, y_train)

            tr_acc = f"{accuracy_score(y_train, grid.predict(X_train)) * 100:.2f}%"
            ts_acc = f"{accuracy_score(y_test, grid.predict(X_test)) * 100:.2f}%"

            print(f"SNKTH-SVC {HV_DIMENTION}, {c}, {tr_acc}, {ts_acc}")

            del grid






        del cb
        del hvs
        del gvfa_model
        # del full_ev_ds

        

        

        # del clf


if __name__ == "__main__":
    main()
