import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import RidgeClassifier, LogisticRegression, RidgeClassifierCV
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split

DATASET_NAME = "NMNIST"




# ---------- 1) Load ----------
if DATASET_NAME == "NMNIST":
    train = np.load("data/NMNIST_hv/train_100_r.npz")
    test  = np.load("data/NMNIST_hv/test_100_r.npz")
elif DATASET_NAME == "ASLDVS":
    train = np.load("data/NMNIST_hv/train_asldvs_100.npz")
    test  = np.load("data/NMNIST_hv/test_asldvs_100.npz")
elif DATASET_NAME == "SNKTH":
    train = np.load("data/NMNIST_hv/train_snkth_100.npz")
    test  = np.load("data/NMNIST_hv/test_snkth_100.npz")
elif DATASET_NAME == "NCARS":
    train = np.load("data/NMNIST_hv/train_ncars_100.npz")
    test  = np.load("data/NMNIST_hv/test_ncars_100.npz")

header = ("Dataset_name,Model,Org_DIM,Gamma,RBF_CONVERTED_DIM,CV,Min_alpha,Max_alpha,Org_Best_alpha,RBF_best_alpha,Train,Test,RBF_Train,RBF_Test")
result_path = Path(f"Results/RBF_analysis_version2_{DATASET_NAME}.csv")



X_train_100_org, Y_train_100_org = train["X"], train["y"]
X_test_100_org, Y_test_100_org = test["X"],  test["y"]

if DATASET_NAME == "NMNIST":
    Xtr_full, ytr_full = X_train_100_org, Y_train_100_org.ravel() 

    X_train_100_org, Xtr_drop, Y_train_100_org, ytr_drop = train_test_split(
    Xtr_full, ytr_full,
    train_size=0.50,          # keep 30% of training
    stratify=ytr_full,        # preserve class balance
    random_state=0            # reproducible
)

Y_train_100_rbf = Y_train_100_org
Y_test_100_rbf = Y_test_100_org

print("Dataset Name - ", {DATASET_NAME})
print("Training-Num_samples- ", len(X_train_100_org))
print("testing-Num samples- ", len(X_test_100_org))
print("Training-classes- ",np.unique(Y_train_100_org).size)
print("Testing-classes- ",np.unique(Y_test_100_org).size)

HV_DIMENTION = 1000
# m = 4096



print(header)
Dimentsions = [4000, 8000]
gammas = 2.0 ** np.arange(-5, 5)
for dim in Dimentsions:
    for gamma in gammas:
        rff = RBFSampler(gamma=gamma, n_components=dim, random_state=0)
        X_train_100_rbf = rff.fit_transform(X_train_100_org)
        X_test_100_rbf = rff.transform(X_test_100_org)


        alphas = 10.0 ** np.arange(-8, 5)
        clf = RidgeClassifierCV(alphas = alphas)
        clf.fit(X_train_100_org, Y_train_100_org)
        tr_acc_ridgeCV = f"{accuracy_score(Y_train_100_org, clf.predict(X_train_100_org)) * 100:.2f}%"
        ts_100_ridgeCV = f"{accuracy_score(Y_test_100_org, clf.predict(X_test_100_org)) * 100:.2f}%"
        org_alpha = clf.alpha_
        del clf

        ## Dor for RBF converted data  ###############################
        clf = RidgeClassifierCV(alphas = alphas, cv=5)
        clf.fit(X_train_100_rbf, Y_train_100_rbf)
        tr_acc_ridgeCV_rbf = f"{accuracy_score(Y_train_100_rbf, clf.predict(X_train_100_rbf)) * 100:.2f}%"
        ts_100_ridgeCV_rbf = f"{accuracy_score(Y_test_100_rbf, clf.predict(X_test_100_rbf)) * 100:.2f}%"
        rbf_alpha = clf.alpha_
        del clf

        line2 = f"{DATASET_NAME},RidgeCV,{1000},{gamma},{dim},{5},{alphas[0]},{alphas[-1]},{org_alpha},{rbf_alpha},{tr_acc_ridgeCV},{ts_100_ridgeCV},{tr_acc_ridgeCV_rbf},{ts_100_ridgeCV_rbf}"
        
        print(line2)
        write_header = (not result_path.exists()) or (result_path.stat().st_size == 0)
        with result_path.open("a+", encoding="utf-8") as f:
            if write_header:
                f.write(header + "\n")
            f.write(line2 + "\n")
