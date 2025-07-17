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

def main(normalized_feat, num_of_graph_events):
    print("[LOG] - parameter initialization.")
    # GRAPH parameters
    DATA_PATH = "data"
    DATASET = "full"  # full / test      size of dataset loading for training and testing

    NORMALIZE_FEAT = False
    NUM_OF_GRAPH_EVENTS = 50  # None, 10, 50, 100. etc
    R = 4
    D_MAX = 16

    # NOISE parameters
    NOICE_REMOVED = True
    NR_BIN_XY_SIZE = 15
    NR_TIME_BIN_SIZE = 20_000
    NR_MINIMUM_EVENTS = 3

    # GVFA parameters
    HV_DIMENTION = 5000
    LAYERS = 5
    DELTA = 1  # 2
    EQUATION = 11
    DEVICE = torch.device("cpu")



    # load event streams
    print("[LOG] - Loading events")
    full_ev_ds = ev_loader(root=DATA_PATH, dataset=DATASET)

    print("[LOG] - Making class objects.")
    MNISTGraph_model = NMNISTGraphDataset(tonic_raw_dataset=full_ev_ds, num_of_graph_events=NUM_OF_GRAPH_EVENTS,
                                          R=R, Dmax=D_MAX,
                                          noise_remove=NOICE_REMOVED, normalized_feat=NORMALIZE_FEAT,
                                          nr_bin_xy_size=NR_BIN_XY_SIZE, nr_minimum_events=NR_MINIMUM_EVENTS,
                                          nr_time_bin_size=NR_TIME_BIN_SIZE)

    gvfa_model = GraphCNN(input_dim=HV_DIMENTION, num_layers=LAYERS, delta=DELTA, graph_pooling_type="sum",
                          neighbor_pooling_type="sum", device=DEVICE, equation=EQUATION).to(DEVICE)
    cb = CodeBook(dim=HV_DIMENTION)
    hvs = HVs(codebook=cb, gvfa_model=gvfa_model)


    X = []
    Y = []
    print("[LOG] - Loading graph and converting to HVs.")
    for i in range(len(full_ev_ds)):
        print(i)
        g = MNISTGraph_model.get(i)
        x, y = hvs.make_hvs(graph=g)
        X.append(x)
        Y.append(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.20, random_state=42
    )

    del cb
    del hvs
    del gvfa_model
    del full_ev_ds

    print("[LOG] - Classification.")
    clf = LogisticRegression(
        solver='saga',  # handles high-dim sparse data efficiently
        penalty='l2',  # ridge regularisation
        max_iter=1000,  # increase if it doesn’t converge
        n_jobs=-1,  # parallelise over cores
        random_state=42
    )
    '''
    clf = XGBClassifier(
        objective='multi:softprob',
        num_class=len(set(y_train)),
        n_estimators=300,       # increase number of trees
        learning_rate=0.05,     # smaller learning rate
        max_depth=8,            # allow deeper trees
        subsample=0.9,          # use 90% of data per tree
        colsample_bytree=0.8,   # use 80% of features per tree
        reg_lambda=1,           # L2 regularization
        reg_alpha=0.5,          # L1 regularization
        n_jobs=-1,
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    ### Grid search
    xgb_model = XGBClassifier(
        objective='multi:softmax',
        num_class=len(np.unique(y_train)),  # number of classes (e.g., 10)
        use_label_encoder=False,
        n_jobs=-1,
        eval_metric='mlogloss',
        random_state=42
    )
    param_grid = {
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=3,  # 3-fold cross-validation
        scoring='accuracy',
        verbose=2,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    clf = grid_search.best_estimator_
    '''

    # clf = RidgeClassifier(alpha=0.001)
    clf.fit(X_train, y_train)
    # 3) evaluate
    print("[LOG] - Predicting")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc * 100:.2f}%\n")

    print("[LOG] - Making Report")
    print("Detailed classification report:")
    print(classification_report(y_test, y_pred))

    print("[LOG]- NUM_OF_GRAPH_EVENTS:", NUM_OF_GRAPH_EVENTS, " | DATASET:", DATASET,
          " | NORMALIZE_FEAT:", NORMALIZE_FEAT,
          " | R:", R, " | D_MAX: ", D_MAX, " | NOICE_REMOVED: ", NOICE_REMOVED,
          " | NR_BIN_XY_SIZE: ", NR_BIN_XY_SIZE, " | NR_TIME_BIN_SIZE: ", NR_TIME_BIN_SIZE, " | NR_MINIMUM_EVENTS: ",
          NR_MINIMUM_EVENTS, " | HV_DIMENTION: ", HV_DIMENTION," | LAYERS: ", LAYERS," | DELTA: ", DELTA," | EQUATION: ", EQUATION,)



if __name__ == "__main__":
    main(normalized_feat=False, num_of_graph_events=100)