
from src.loader import graph_loader
from src.graph_to_vec_converter import make_hvs
from sklearn.metrics       import accuracy_score, classification_report
from sklearn.linear_model  import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np
from sklearn.linear_model import RidgeClassifier

def main(normalized_feat, num_of_graph_events):
    graphs = graph_loader(normalized_feat=normalized_feat, num_of_graph_events=num_of_graph_events)
    X, Y = make_hvs(graphs, normalized_feat=normalized_feat, num_of_graph_events=num_of_graph_events)

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.20, random_state=42
    )

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
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc * 100:.2f}%\n")

    print("Detailed classification report:")
    print(classification_report(y_test, y_pred))



if __name__ == "__main__":
    main(normalized_feat=False, num_of_graph_events=100)