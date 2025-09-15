import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import RidgeClassifier, LogisticRegression, RidgeClassifierCV
import joblib
from pathlib import Path

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

header = ("Dataset_name,Model,Org_DIM,Gamma,RBF_CONVERTED_DIM,C_value,Train,Test,RBF_Train,RBF_Test")
result_path = Path("Results/RBF_analysis.csv")

X_train_100_org, Y_train_100_org = train["X"], train["y"]
X_test_100_org, Y_test_100_org = test["X"],  test["y"]

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
gammas = [0.0001, 0.001, 0.01, 1, 10]
for dim in Dimentsions:
    for gamma in gammas:
        rff = RBFSampler(gamma=gamma, n_components=dim, random_state=0)
        X_train_100_rbf = rff.fit_transform(X_train_100_org)
        X_test_100_rbf = rff.transform(X_test_100_org)

        tr_acc_lsvc_list = []
        ts_100_lsvc_list = []
        tr_acc_svc_list = []
        ts_100_svc_list = []
        tr_acc_lsvc_rbf_list = []
        ts_100_lsvc_rbf_list = []


        CS = [0.0001, 0.001, 0.1, 1, 3,  5, 7, 10, 100, 1000]

        # print(f"\n\n\nOriginal_HV_Dimention {1000}")
        # print(f"RBF Converted Dimention {dim}")
        # print(f"Gamma {gamma}")

        for i in range  (len(CS)):
            c = CS[i]
            ##### ORG data########
            clf = LinearSVC(C=c, class_weight='balanced', max_iter=20000)
            clf.fit(X_train_100_org, Y_train_100_org)
            tr_acc_lsvc = f"{accuracy_score(Y_train_100_org, clf.predict(X_train_100_org)) * 100:.2f}%"
            ts_100_lsvc = f"{accuracy_score(Y_test_100_org, clf.predict(X_test_100_org)) * 100:.2f}%"
            # print(f"{DATASET_NAME}-LinearSVC-ORG {HV_DIMENTION}, {c}, {tr_acc}, {ts_100}")
            del clf

            clf = SVC(kernel="rbf", C=c, gamma='scale', class_weight="balanced")
            clf.fit(X_train_100_org, Y_train_100_org)
            tr_acc_svc = f"{accuracy_score(Y_train_100_org, clf.predict(X_train_100_org)) * 100:.2f}%"
            ts_100_svc = f"{accuracy_score(Y_test_100_org, clf.predict(X_test_100_org)) * 100:.2f}%"
            # print(f"{DATASET_NAME}-SVC-ORG {HV_DIMENTION}, {c}, {tr_acc}, {ts_100}")
            del clf


            ######## RBF data ##########
            clf = LinearSVC(C=c, class_weight='balanced', max_iter=20000)
            clf.fit(X_train_100_rbf, Y_train_100_rbf)
            tr_acc_lsvc_rbf = f"{accuracy_score(Y_train_100_rbf, clf.predict(X_train_100_rbf)) * 100:.2f}%"
            ts_100_lsvc_rbf = f"{accuracy_score(Y_test_100_rbf, clf.predict(X_test_100_rbf)) * 100:.2f}%"
            # print(f"{DATASET_NAME}-LinearSVC-RBF {HV_DIMENTION}, {c}, {tr_acc}, {ts_100}")
            del clf

            line1 = f"{DATASET_NAME},SVC,{1000},{gamma},{dim},{c},{tr_acc_svc},{ts_100_svc},_,_"
            line2 = f"{DATASET_NAME},Linear_SVC,{1000},{gamma},{dim},{c},{tr_acc_lsvc},{ts_100_lsvc},{tr_acc_lsvc_rbf},{ts_100_lsvc_rbf}"
 
            print(line1)
            print(line2)
            
            write_header = (not result_path.exists()) or (result_path.stat().st_size == 0)
            with result_path.open("a+", encoding="utf-8") as f:
                if write_header:
                    f.write(header + "\n")
                f.write(line1 + "\n")
                f.write(line2 + "\n")



        
        clf = RidgeClassifierCV()
        clf.fit(X_train_100_org, Y_train_100_org)
        tr_acc_ridgeCV = f"{accuracy_score(Y_train_100_org, clf.predict(X_train_100_org)) * 100:.2f}%"
        ts_100_ridgeCV = f"{accuracy_score(Y_test_100_org, clf.predict(X_test_100_org)) * 100:.2f}%"
        # print(f"{DATASET_NAME}-RidgeClassifierCV-ORG {HV_DIMENTION}, {tr_acc}, {ts_100}")
        del clf

        clf = RidgeClassifier()
        clf.fit(X_train_100_org, Y_train_100_org)
        tr_acc_ridge = f"{accuracy_score(Y_train_100_org, clf.predict(X_train_100_org)) * 100:.2f}%"
        ts_100_ridge = f"{accuracy_score(Y_test_100_org, clf.predict(X_test_100_org)) * 100:.2f}%"
        # print(f"{DATASET_NAME}-RidgeClassifier-ORG {HV_DIMENTION}, {tr_acc}, {ts_100}")
        del clf

        clf = LogisticRegression(
            solver='saga',  # handles high-dim sparse data efficiently
            penalty='l2',  # ridge regularisation
            max_iter=1000,  # increase if it doesn’t converge
            n_jobs=-1,  # parallelise over cores
            random_state=42
        )
        clf.fit(X_train_100_org, Y_train_100_org)
        tr_acc_logic = f"{accuracy_score(Y_train_100_org, clf.predict(X_train_100_org)) * 100:.2f}%"
        ts_100_logic = f"{accuracy_score(Y_test_100_org, clf.predict(X_test_100_org)) * 100:.2f}%"
        # print(f"{DATASET_NAME}-logistic-ORG {HV_DIMENTION}, {tr_acc}, {ts_100}")
        del clf

        ## Dor for RBF converted data  ###############################
        clf = RidgeClassifierCV()
        clf.fit(X_train_100_rbf, Y_train_100_rbf)
        tr_acc_ridgeCV_rbf = f"{accuracy_score(Y_train_100_rbf, clf.predict(X_train_100_rbf)) * 100:.2f}%"
        ts_100_ridgeCV_rbf = f"{accuracy_score(Y_test_100_rbf, clf.predict(X_test_100_rbf)) * 100:.2f}%"
        # print(f"{DATASET_NAME}-RidgeClassifierCV-RBF {HV_DIMENTION}, {tr_acc}, {ts_100}")
        del clf

        clf = RidgeClassifier()
        clf.fit(X_train_100_rbf, Y_train_100_rbf)
        tr_acc_ridge_rbf = f"{accuracy_score(Y_train_100_rbf, clf.predict(X_train_100_rbf)) * 100:.2f}%"
        ts_100_ridge_rbf = f"{accuracy_score(Y_test_100_rbf, clf.predict(X_test_100_rbf)) * 100:.2f}%"
        # print(f"{DATASET_NAME}-RidgeClassifier-RBF {HV_DIMENTION}, {tr_acc}, {ts_100}")
        del clf

        clf = LogisticRegression(
            solver='saga',  # handles high-dim sparse data efficiently
            penalty='l2',  # ridge regularisation
            max_iter=1000,  # increase if it doesn’t converge
            n_jobs=-1,  # parallelise over cores
            random_state=42
        )
        clf.fit(X_train_100_rbf, Y_train_100_rbf)
        tr_acc_logic_rbf = f"{accuracy_score(Y_train_100_rbf, clf.predict(X_train_100_rbf)) * 100:.2f}%"
        ts_100_logic_rbf = f"{accuracy_score(Y_test_100_rbf, clf.predict(X_test_100_rbf)) * 100:.2f}%"
        # print(f"{DATASET_NAME}-logistic-RBF {HV_DIMENTION}, {tr_acc}, {ts_100}")
        del clf

        line1 = f"{DATASET_NAME},Ridge,{1000},{gamma},{dim},_,{tr_acc_ridge},{ts_100_ridge},{tr_acc_ridge_rbf},{ts_100_ridge_rbf}"
        line2 = f"{DATASET_NAME},RidgeCV,{1000},{gamma},{dim},_,{tr_acc_ridgeCV},{ts_100_ridgeCV},{tr_acc_ridgeCV_rbf},{ts_100_ridgeCV_rbf}"
        line3 = f"{DATASET_NAME},Logistic,{1000},{gamma},{dim},_,{tr_acc_logic},{ts_100_logic},{tr_acc_logic_rbf},{ts_100_logic_rbf}"

        print(line1)
        print(line2)
        print(line3)
        write_header = (not result_path.exists()) or (result_path.stat().st_size == 0)
        with result_path.open("a+", encoding="utf-8") as f:
            if write_header:
                f.write(header + "\n")
            f.write(line1 + "\n")
            f.write(line2 + "\n")
            f.write(line3 + "\n")