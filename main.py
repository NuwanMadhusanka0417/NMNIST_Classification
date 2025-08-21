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
    NUM_OF_GRAPH_EVENTS = None  # None, 10, 50, 100. etc
    
    R = 7
    D_MAX = 30

    # NOISE parameters
    NOICE_REMOVED = True
    NR_BIN_XY_SIZE = 4
    NR_TIME_BIN_SIZE = 20_000
    NR_MINIMUM_EVENTS = 3

    # GVFA parameters
    HV_DIMENTION = 15000
    LAYERS = 5
    DELTA = 1  # 2
    EQUATION = 11
    DEVICE = torch.device("cpu")

    # load event streams
    print("[LOG] - Loading events")
    # full_ev_ds = ev_loader(root=DATA_PATH, dataset=DATASET)
    ds = ev_loader(root=DATA_PATH, dataset=DATASET)
    train_ds, test_ds = train_test_split(ds, test_size=0.2, random_state=10, shuffle=True)  
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

    items = [7000, 10000, 15000]
    for item in items:
        HV_DIMENTION = item
        gvfa_model = GraphCNN(input_dim=HV_DIMENTION, num_layers=LAYERS, delta=DELTA, graph_pooling_type="sum",
                            neighbor_pooling_type="sum", device=DEVICE, equation=EQUATION).to(DEVICE)
        cb = CodeBook(dim=HV_DIMENTION, x_max=X_MAX, y_max=Y_MAX, t_max=T_MAX, t_step=T_STEP)
        hvs = HVs(codebook=cb, gvfa_model=gvfa_model)


        X_train, X_test, y_train, y_test = [],[],[],[]
        print("[LOG] - Loading graph and converting to HVs.")
        for i in range(len(train_ds)):
            g = MNISTGraph_model_train.get(i)
            x, y = hvs.make_hvs(graph=g)
            X_train.append(x)
            y_train.append(y)
        
        for i in range(len(test_ds)):
            g = MNISTGraph_model_test.get(i)
            x, y = hvs.make_hvs(graph=g)
            X_test.append(x)
            y_test.append(y)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42, shuffle=True)
        del cb
        del hvs
        del gvfa_model
        # del full_ev_ds

        el = [1, 3, 5]
        for elm in el:
            print(elm)
            print("[LOG] - Classification.")

            clf = SVC(kernel="rbf", C=elm,class_weight="balanced", gamma='scale')  # best in 100 - c = 3, weight = NOne, gamma = scale SVC(kernel="rbf", C=0.1, gamma=0.9,degree=6)
    
            clf.fit(X_train, y_train)

            print(f"Train accuracy: {accuracy_score(y_train, clf.predict(X_train)) * 100:.2f}%")
            print(f"Test  accuracy: {accuracy_score(y_test, clf.predict(X_test)) * 100:.2f}%")

            print("[LOG]- NUM_OF_GRAPH_EVENTS:", NUM_OF_GRAPH_EVENTS, " | DATASET:", DATASET,
                " | NORMALIZE_FEAT:", NORMALIZE_FEAT,
                " | R:", R, " | D_MAX: ", D_MAX, " | NOICE_REMOVED: ", NOICE_REMOVED,
                " | NR_BIN_XY_SIZE: ", NR_BIN_XY_SIZE, " | NR_TIME_BIN_SIZE: ", NR_TIME_BIN_SIZE, " | NR_MINIMUM_EVENTS: ",
                NR_MINIMUM_EVENTS, " | HV_DIMENTION: ", HV_DIMENTION," | LAYERS: ", LAYERS," | DELTA: ", DELTA," | EQUATION: ", EQUATION,)

            del clf

if __name__ == "__main__":
    main()
