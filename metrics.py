import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
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
import warnings

clf_dict = {
    "LR": LogisticRegression,
    "GB": GradientBoostingClassifier,
    "XGB": xgb.XGBClassifier,
    "KNN": KNeighborsClassifier, 
    "NN": MLPClassifier,
}

race_encoding =  {
        1.0: "white",
        2.0: "black",
        3.0: "other",
        6.0: "asian",
    }

race_grouping=  {
        1.0:1.0,
        2.0:2.0,
        3.0:3.0,
        4.0:3.0, # mapping "American Indian Alone" to other
        5.0:3.0, # mapping "Alaska Naive Alone" to other
        6.0:6.0, 
        7.0:3.0, # mapping "Native Hawaiian Alone" to other
        8.0:3.0, # mapping "Some Other Race alone" to other
        9.0:3.0, # mapping "Two or More Races" to other
    }

@ignore_warnings(category=ConvergenceWarning)
def model_choice(clf, xtrain=None, ytrain=None):
    param_grid_nn = {
        "mlp__alpha": [0.01, 0.05, 0.1],
        "mlp__learning_rate": ["constant", "adaptive"],
        'mlp__hidden_layer_sizes': [(8, 2), (12, 3), (16, 4)] 
    }
    param_grid_knn = {
        "knn__n_neighbors": [3, 5, 7]
    }
    if clf == "XBG":
        model = make_pipeline(
            StandardScaler(), clf_dict[clf](objective="binary:logistic")
        )
    elif clf == "KNN":
        model = Pipeline(
            [
                ("scalar", StandardScaler()),
                ("knn", KNeighborsClassifier()),
            ]
        )
        print("running model search")
        grid_search = GridSearchCV(model, param_grid_knn, n_jobs=-1, cv=5)
        
        with warnings.catch_warnings(): 
            warnings.filterwarnings("ignore",category=ConvergenceWarning) 
            grid_search.fit(xtrain, ytrain)
        # final model
        model = Pipeline(
            [
                ("scalar", StandardScaler()),
                ("knn", KNeighborsClassifier(grid_search.best_params_["knn__n_neighbors"])),
            ]
        )
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
        grid_search = GridSearchCV(model, param_grid_nn, n_jobs=-1, cv=5)
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
    group_dict = {}
    vals, counts = np.unique(group_arr, return_counts=True)
    for g, g_count in zip(vals, counts): 
        if g_count > min_size: 
            g_acc = np.mean(correct_arr[group_arr == g])
            group_dict[race_encoding[g]] = g_acc
    # find non-white accuracy
    group_dict["non-white"] = np.mean(correct_arr[group_arr != 1])
    return group_dict

def group_accuracy_ot(target, pred, thresh, group_arr, min_size=10): 
    group_dict = {}
    vals, counts = np.unique(group_arr, return_counts=True)
    for g, g_count in zip(vals, counts): 
        if g_count > min_size: 
            g_acc = accuracy_score(y_true=target[group_arr == g], 
                                   y_pred=(pred > thresh)[group_arr == g])
            group_dict[race_encoding[g]] = g_acc
    # find non-white accuracy
    group_dict["non-white"] = accuracy_score(y_true=target[group_arr != 1], 
                                   y_pred=(pred > thresh)[group_arr != 1])
    return group_dict

def group_f1(target, pred, group_arr, min_size=10): 
    group_dict = {}
    vals, counts = np.unique(group_arr, return_counts=True)
    for g, g_count in zip(vals, counts): 
        if g_count > min_size: 
            g_f1 = f1_score(target[group_arr == g], 
                                      pred[group_arr == g])
            group_dict[race_encoding[g]] = g_f1
    # find non-white f1
    group_dict["non-white"] = f1_score(target[group_arr != 1], 
                                      pred[group_arr != 1])
    return group_dict

def group_auc(target, pred, group_arr, min_size=10): 
    group_dict = {}
    vals, counts = np.unique(group_arr, return_counts=True)
    for g, g_count in zip(vals, counts): 
        if g_count > min_size: 
            # AUC only valid if two classes exist 
            if (len(np.unique(target[group_arr == g])) > 1): 
                g_auc = roc_auc_score(target[group_arr == g], 
                                      pred[group_arr == g])

                group_dict[race_encoding[g]] = g_auc
    # find all non-white AUC                 
    group_dict["non-white"] =roc_auc_score(target[group_arr != 1], 
                                      pred[group_arr != 1])
    
    return group_dict 


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
    data = c.transform(input_data)
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