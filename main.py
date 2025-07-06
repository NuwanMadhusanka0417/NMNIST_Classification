
from src.loader import graph_loader
from src.graph_to_vec_converter import make_hvs
from sklearn.model_selection import train_test_split
from sklearn.metrics       import accuracy_score, classification_report
from sklearn.linear_model  import LogisticRegression


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
    clf.fit(X_train, y_train)

    # 3) evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc * 100:.2f}%\n")

    print("Detailed classification report:")
    print(classification_report(y_test, y_pred))



if __name__ == "__main__":
    main(normalized_feat=False, num_of_graph_events=100)