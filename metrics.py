import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
import xgboost as xgb

from sklearn.decomposition import PCA
import scipy.special as sp
import scipy.stats as st
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

# ignore grid search warnings  
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

clf_dict = {
    "LR": LogisticRegression,
    "GB": GradientBoostingClassifier,
    "XGB": xgb.XGBClassifier,
    "SVM": LinearSVC,
    "NN": MLPClassifier,
}

@ignore_warnings(category=ConvergenceWarning)
def model_choice(clf, xtrain=None, ytrain=None):
    param_grid = {
        "mlp__alpha": [0.01, 0.05, 0.1],
        "mlp__learning_rate": ["constant", "adaptive"],
        'mlp__hidden_layer_sizes': [(8, 2), (12, 3), (16, 4)] 
    }
    if clf == "XBG":
        model = make_pipeline(
            StandardScaler(), clf_dict[clf](objective="binary:logistic")
        )
    elif clf == "SVM":
        model = make_pipeline(StandardScaler(), clf_dict[clf](C=0.2))
    elif clf == "NN":
        model = Pipeline(
            [
                ("scalar", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        solver="sgd",
                        hidden_layer_sizes=(8, 2),
                        random_state=1,
                        max_iter=500,
                    ),
                ),
            ]
        )
        print("running model search")
        grid_search = GridSearchCV(model, param_grid, n_jobs=-1, cv=5)
        grid_search.fit(xtrain, ytrain)
        # final model
        model = Pipeline(
            [
                ("scalar", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        solver="sgd",
                        hidden_layer_sizes=grid_search.best_params_["mlp__hidden_layer_sizes"],
                        random_state=1,
                        max_iter=500,
                        alpha=grid_search.best_params_["mlp__alpha"],
                        learning_rate=grid_search.best_params_["mlp__learning_rate"],
                    ),
                ),
            ]
        )
    else:
        model = make_pipeline(StandardScaler(), clf_dict[clf]())
    return model


def group_accuracy(correct_arr, group_arr, min_size=10): 
    g_acc_arr = [] 
    vals, counts = np.unique(group_arr, return_counts=True)
    for g, g_count in zip(vals, counts): 
        if g_count > min_size: 
            g_acc = np.mean(correct_arr[group_arr == g])
            g_acc_arr.append(g_acc)
    
    group_bin = {}
    group_bin["white"] = np.mean(correct_arr[group_arr == 1])
    group_bin["black"] = np.mean(correct_arr[group_arr == 2])
    group_bin["non-white"] = np.mean(correct_arr[group_arr != 1])
    return g_acc_arr, group_bin

def group_f1(target, pred, group_arr, min_size=10): 
    g_f1_arr = [] 
    vals, counts = np.unique(group_arr, return_counts=True)
    for g, g_count in zip(vals, counts): 
        if g_count > min_size: 
            g_f1 = f1_score(target[group_arr == g], 
                                      pred[group_arr == g])
            g_f1_arr.append(g_f1)
    group_bin = {}
    group_bin["white"] = f1_score(target[group_arr == 1], 
                                      pred[group_arr == 1])
    group_bin["black"] = f1_score(target[group_arr == 2], 
                                      pred[group_arr == 2])
    group_bin["non-white"] = f1_score(target[group_arr != 1], 
                                      pred[group_arr != 1])
    return g_f1_arr, group_bin

def group_auc(target, pred, group_arr, min_size=10): 
    g_auc_arr = [] 
    vals, counts = np.unique(group_arr, return_counts=True)
    for g, g_count in zip(vals, counts): 
        if g_count > min_size: 
            # AUC only valid if two classes exist 
            if (len(np.unique(target[group_arr == g])) > 1) and \
            (len(np.unique(pred[group_arr == g])) > 1): 
                g_auc = roc_auc_score(target[group_arr == g], 
                                      pred[group_arr == g])

                g_auc_arr.append(g_auc)
        
    group_bin = {}
    group_bin["white"] = roc_auc_score(target[group_arr == 1], 
                                      pred[group_arr == 1])
    group_bin["black"] = roc_auc_score(target[group_arr == 2], 
                                      pred[group_arr == 2])
    group_bin["non-white"] =roc_auc_score(target[group_arr != 1], 
                                      pred[group_arr != 1])
    
    return g_auc_arr, group_bin 


def init_density_scale(input_data, n_components=3): 
    cx = make_pipeline(StandardScaler(), PCA(n_components=n_components)) 
    
    data = cx.fit_transform(input_data)
    params = {"bandwidth": np.logspace(-1, 10, 20)}
    grid = GridSearchCV(KernelDensity(), params)
    grid.fit(data)
    kde = grid.best_estimator_
    print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
    kde.fit(data)
    return cx, kde

def init_density(input_data, c):
    data = c.fit_transform(input_data)
    params = {"bandwidth": np.logspace(-1, 10, 20)}
    grid = GridSearchCV(KernelDensity(), params)
    grid.fit(data)
    kde = grid.best_estimator_
    #print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
    kde.fit(data)
    return kde


def kl_input(x, pkde, qkde, c): 
    px = np.exp(pkde.score_samples(c.transform(x)))
    qx = np.exp(qkde.score_samples(c.transform(x)))
    return sp.kl_div(px, qx).mean()


def entropy_input(x, pkde, qkde, c): 
    px = np.exp(pkde.score_samples(c.transform(x)))
    qx = np.exp(qkde.score_samples(c.transform(x)))
    return st.entropy(px, qx)
    
def hello(): 
    print("you are doing great")