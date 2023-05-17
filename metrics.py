import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
import scipy.special as sp
import scipy.stats as st
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def group_accuracy(correct_arr, group_arr, min_size=10): 
    g_acc_arr = [] 
    vals, counts = np.unique(group_arr, return_counts=True)
    for g, g_count in zip(vals, counts): 
        if g_count > min_size: 
            g_acc = np.mean(correct_arr[group_arr == g])
            g_acc_arr.append(g_acc)
    return g_acc_arr 


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