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
    full_ev_ds = ev_loader(dataset=DATASET)

    ###################################################################

    # print(full_ev_ds.shape)

    indices = np.arange(len(full_ev_ds))
    labels  = [full_ev_ds[i][1] for i in indices]
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=labels
    )

    ds_train = Subset(full_ev_ds, train_idx.tolist())
    ds_test  = Subset(full_ev_ds, test_idx.tolist())

    # sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    # ds_train, ds_test= next(sgkf.split(np.zeros(len(y)), y, groups=recording_id))

    # ds_train, ds_test = train_test_split(full_ev_ds, test_size=0.2, random_state=42, shuffle=True, )   
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
    
    HV_Dimensions = [5000, 10000, 15000]
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
        print("Start Classification")
        CS = [0.05]
        iteration = [100]
        for iters in iteration:
            for c in CS:
                # print("############  C = ", c, "   iteration = ", iters)
                # clf = Pipeline([
                #     ("svd", TruncatedSVD(n_components=512, random_state=42)),
                #     ("scaler", StandardScaler()),  # after SVD output is dense, mean-centering is fine
                #     ("clf", LogisticRegression(
                #         solver="saga",
                #         penalty="elasticnet",      # better than pure L2 for high-dim; switch to 'l2' if you prefer
                #         l1_ratio=0.15,             # ignored if penalty='l2'
                #         multi_class="multinomial",
                #         C=c,
                #         max_iter=iters,
                #         n_jobs=-1,
                #         random_state=42
                #     ))
                # ])

                # clf = SVC(kernel="rbf", C=c, gamma=0.9,degree=6)
                clf = LogisticRegression(solver = 'saga', penalty = 'l2', n_jobs=-1, random_state=42)

                # svm = Pipeline([
                #     ("svd", TruncatedSVD(n_components=512, random_state=42)),
                #     ("scaler", StandardScaler()),
                #     ("svm", LinearSVC(C=c, max_iter=iters, random_state=42))
                # ])

                clf.fit(X_train_100, Y_train_100)
                # svm.fit(X_train_100, Y_train_100)


                #### first classofoer###########
                acc_100_tr_c = f"{accuracy_score(Y_train_100, clf.predict(X_train_100)) * 100:.2f}"
                acc_100_c = f"{accuracy_score(Y_test_100, clf.predict(X_test_100)) * 100:.2f}"
                acc_50_c = f"{accuracy_score(y_test_50, clf.predict(X_test_50)) * 100:.2f}"
                acc_10_c = f"{accuracy_score(y_test_10, clf.predict(X_test_10)) * 100:.2f}"

                ### second classofoer###########
                # acc_100_tr_s = f"{accuracy_score(Y_train_100, svm.predict(X_train_100)) * 100:.2f}"
                # acc_100_s = f"{accuracy_score(Y_test_100, svm.predict(X_test_100)) * 100:.2f}"
                # acc_50_s = f"{accuracy_score(y_test_50, svm.predict(X_test_50)) * 100:.2f}"
                # acc_10_s = f"{accuracy_score(y_test_10, svm.predict(X_test_10)) * 100:.2f}"


                print("---------------------------------")
                # print("Dimention, C, Iterations, 100_TR, 100, 50, 10")
                line1 = f"SVC, {HV_DIMENTION}, {c}, {iters}, {acc_100_tr_c},  {acc_100_c}, {acc_50_c}, {acc_10_c} \n" 
                print(line1)
                # line2 = f"svm, {HV_DIMENTION}, {c}, {iters}, {acc_100_tr_s},  {acc_100_s}, {acc_50_s}, {acc_10_s} \n" 
                # print(line2)

                with open(out_path, "a+", encoding="utf-8") as f:
                    f.write(line1)
                    # f.write(line2)
                '''print("[LOG]- First - NUM_OF_GRAPH_EVENTS:", NUM_OF_GRAPH_EVENTS, " | DATASET:", DATASET,
                    " | NORMALIZE_FEAT:", NORMALIZE_FEAT,
                    " | R:", R, " | D_MAX: ", D_MAX, " | NOICE_REMOVED: ", NOICE_REMOVED,
                    " | NR_BIN_XY_SIZE: ", NR_BIN_XY_SIZE, " | NR_TIME_BIN_SIZE: ", NR_TIME_BIN_SIZE,
                    " | NR_MINIMUM_EVENTS: ",
                    NR_MINIMUM_EVENTS, " | HV_DIMENTION: ", HV_DIMENTION, " | LAYERS: ", LAYERS, " | DELTA: ", DELTA,
                    " | EQUATION: ", EQUATION, )'''

                del clf
                # del svm


        
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
