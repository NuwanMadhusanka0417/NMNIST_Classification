from src.graph_to_vec_converter import  HVs
from sklearn.metrics       import accuracy_score, classification_report
from sklearn.linear_model  import LogisticRegression, RidgeClassifierCV, RidgeClassifier
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
import numpy as np
from torch.utils.data import Subset
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.pipeline import Pipeline
from datetime import datetime
import os
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.svm  import SVC
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.kernel_approximation import RBFSampler, Nystroem


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


    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = "logs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"clf_svm_{ts}.txt")

    # load event streams
    print("[LOG] - Loading events")
    full_ev_ds = ev_loader(root=DATA_PATH, dataset=DATASET)

    ###################################################################

    indices = np.arange(len(full_ev_ds))
    labels  = [full_ev_ds[i][1] for i in indices]
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        train_size=0.8, 
        random_state=42,
        shuffle=True,
        stratify=labels
    )

    ds_train = Subset(full_ev_ds, train_idx.tolist())
    ds_test  = Subset(full_ev_ds, test_idx.tolist())

    # ds_train, ds_test = train_test_split(full_ev_ds, test_size=0.2, random_state=10, shuffle=True)   
    ######################################################################
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
    
    HV_Dimensions = [1000] #, 5000, 10000]
    print("Start For loop")
    for HV_DIMENTION in HV_Dimensions:

        gvfa_model = GraphCNN(input_dim=HV_DIMENTION, num_layers=LAYERS, delta=DELTA, graph_pooling_type="sum",
                          neighbor_pooling_type="sum", device=DEVICE, equation=EQUATION).to(DEVICE)
        cb = CodeBook(dim=HV_DIMENTION)
        hvs = HVs(codebook=cb, gvfa_model=gvfa_model)

        X_train_100, X_test_100, X_test_50, X_test_10, Y_train_100, Y_test_100, y_test_50, y_test_10 = [], [], [], [], [], [], [], []
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

            x_50, y_50 = hvs.make_hvs(graph=g_50)
            x_10, y_10 = hvs.make_hvs(graph=g_10)

            X_test_50.append(x_50)
            X_test_10.append(x_10)
            y_test_50.append(y_50)
            y_test_10.append(y_10)


        scaler = StandardScaler()
        X_train_100_r = scaler.fit_transform(X_train_100)
        X_test_100_r = scaler.transform(X_test_100)
        X_test_50_r = scaler.fit_transform(X_test_50)
        X_test_10_r = scaler.fit_transform(X_test_10)

        np.savez_compressed("data/NMNIST_hv/train_100_r.npz", X=X_train_100_r, y=Y_train_100)
        np.savez_compressed("data/NMNIST_hv/test_100_r.npz",  X=X_test_100_r,  y=Y_test_100)

        print("Dtaset details")

        print("train data set - ", len(X_train_100_r))
        print("test data set - ", len(X_test_100_r))

        Ms = [4096, 2048, 1024]
        for m in Ms:

            print("M = ", m)
            rff = RBFSampler(gamma=0.001, n_components=m, random_state=0)
            X_train_100 = rff.fit_transform(X_train_100_r)
            X_test_100 = rff.transform(X_test_100_r)
            X_test_50 = rff.transform(X_test_50_r)
            X_test_10 = rff.transform(X_test_10_r)


            print("Start Classification")

            CS = [3,4, 5,6,7, 9]
            for c in CS:
                # clf = SVC(kernel="rbf", C=c, gamma='scale', class_weight="balanced")
                clf = LinearSVC(C=c, class_weight='balanced', max_iter=20000)

                clf.fit(X_train_100, Y_train_100)

                tr_acc = f"{accuracy_score(Y_train_100, clf.predict(X_train_100)) * 100:.2f}%"
                ts_100 = f"{accuracy_score(Y_test_100, clf.predict(X_test_100)) * 100:.2f}%"
                ts_50 =  f"{accuracy_score(y_test_50, clf.predict(X_test_50)) * 100:.2f}%"
                ts_10 = f"{accuracy_score(y_test_10, clf.predict(X_test_10)) * 100:.2f}%"

                print(f"NMNST-LinearSVC {HV_DIMENTION}, {c}, {tr_acc}, {ts_100}, {ts_50}, {ts_10}")

                del clf

                clf = SVC(kernel="rbf", C=c, gamma='scale', class_weight="balanced")

                clf.fit(X_train_100_r, Y_train_100)

                tr_acc = f"{accuracy_score(Y_train_100, clf.predict(X_train_100_r)) * 100:.2f}%"
                ts_100 = f"{accuracy_score(Y_test_100, clf.predict(X_test_100_r)) * 100:.2f}%"
                ts_50 =  f"{accuracy_score(y_test_50, clf.predict(X_test_50_r)) * 100:.2f}%"
                ts_10 = f"{accuracy_score(y_test_10, clf.predict(X_test_10_r)) * 100:.2f}%"

                print(f"NMNST-SVC {HV_DIMENTION}, {c}, {tr_acc}, {ts_100}, {ts_50}, {ts_10}")

                del clf


            clf = RidgeClassifierCV()

            clf.fit(X_train_100, Y_train_100)

            tr_acc = f"{accuracy_score(Y_train_100, clf.predict(X_train_100)) * 100:.2f}%"
            ts_100 = f"{accuracy_score(Y_test_100, clf.predict(X_test_100)) * 100:.2f}%"
            ts_50 =  f"{accuracy_score(y_test_50, clf.predict(X_test_50)) * 100:.2f}%"
            ts_10 = f"{accuracy_score(y_test_10, clf.predict(X_test_10)) * 100:.2f}%"

            print(f"NMNST- RidgeClassifierCV {HV_DIMENTION}, {tr_acc}, {ts_100}, {ts_50}, {ts_10}")

            del clf

            clf = RidgeClassifier()

            clf.fit(X_train_100, Y_train_100)

            tr_acc = f"{accuracy_score(Y_train_100, clf.predict(X_train_100)) * 100:.2f}%"
            ts_100 = f"{accuracy_score(Y_test_100, clf.predict(X_test_100)) * 100:.2f}%"
            ts_50 =  f"{accuracy_score(y_test_50, clf.predict(X_test_50)) * 100:.2f}%"
            ts_10 = f"{accuracy_score(y_test_10, clf.predict(X_test_10)) * 100:.2f}%"

            print(f"NMNST- RidgeClassifier {HV_DIMENTION}, {tr_acc}, {ts_100}, {ts_50}, {ts_10}")

            del clf
            
            clf = LogisticRegression(
                solver='saga',  # handles high-dim sparse data efficiently
                penalty='l2',  # ridge regularisation
                max_iter=1000,  # increase if it doesn’t converge
                n_jobs=-1,  # parallelise over cores
                random_state=42
            )

            clf.fit(X_train_100, Y_train_100)

            tr_acc = f"{accuracy_score(Y_train_100, clf.predict(X_train_100)) * 100:.2f}%"
            ts_100 = f"{accuracy_score(Y_test_100, clf.predict(X_test_100)) * 100:.2f}%"
            ts_50 =  f"{accuracy_score(y_test_50, clf.predict(X_test_50)) * 100:.2f}%"
            ts_10 = f"{accuracy_score(y_test_10, clf.predict(X_test_10)) * 100:.2f}%"

            print(f"NMNST- logistic {HV_DIMENTION}, {tr_acc}, {ts_100}, {ts_50}, {ts_10}")

            del clf

            del gvfa_model
            del cb
            del hvs
            del X_train_100
            del X_test_100
            del X_test_50
            del X_test_10
            del Y_train_100
            del Y_test_100
            del y_test_50
            del y_test_10
        


if __name__ == "__main__":
    main(normalized_feat=False, num_of_graph_events=100)