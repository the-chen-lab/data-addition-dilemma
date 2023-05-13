# similar to run_mimic
# python run_sequential.py --data yelp --model lr --n 1000 (and etc)

import numpy as np
import os.path
import pandas as pd
import pickle

from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import xgboost as xgb


pd.set_option("display.max_rows", None)



def get_mimic_sequential_data():
    """
    output
     - X: numpy array or csr matrix
     - y: np array of outcomes
     - years:
     - groups: 
    """
    
def get_avail_idx(num, year_idx_dict, ignore_idx, avail_years):
    """
    input:
     - num: int of samples needed
     - year_idx_dict: dict of {year: list of idx} for the df
     - ignore_idx: list of idx to ignore
     - avail_years: list of years that are considered
    
    output:
     - avail_dict: dict of {num to sample: list of valid idx}
    """
    year_idx_dict_ignore = {k:[i for i in v if i not in ignore_idx] for k,v in year_idx_dict.items()}
    avail_dict = {}
    remaining = num
    for year in sorted(year_idx_dict_ignore.keys()):
        if year in avail_years and remaining > 0:
            avail_year_idx = year_idx_dict_ignore[year]
            avail_year_num_idx = len(avail_year_idx)
            
            use_this_year = min(remaining, avail_year_num_idx)
            avail_dict[use_this_year] = avail_year_idx
            remaining -= avail_year_num_idx
    return avail_dict

def get_sample_avail_idx(avail_dict):
    avail_idx_lst = list()
    
    for num_to_sample, lst_to_sample in avail_dict.items():
        sample_idx = np.random.choice(lst_to_sample, size=num_to_sample, replace=False)
        avail_idx_lst.append(sample_idx)
        
    avail_idx = np.concatenate(avail_idx_lst)
    return avail_idx

def run_sequential(X,y,years,groups,model_name,train_N, data_name, train_set_size=100):
    """
    Main function for sequential data experiments
    
    X: np array or csr matrix
    y: binary outcome values
    years: lst of years, assumed int
    groups: lst sensitive attributes, e.g., race
    model_name: lr, nn, svm, or xgb
    train_N: number of training set size
    data_name: yelp, mimic, folk
    """
    
#     test_set_size = 100
    N = X.shape[0]
    
    year_counts_dict = pd.Series(years).to_dict()
    year_idx_dict = {year: np.where(years == year)[0] for year in np.unique(years)}
    
    max_year = max(years)
    # TODO: get ref/test idx if not already cached
    
    ref_idx_fname = '%s_ref_idx.pk' % data_name
    gen_idx_fname = '%s_gen_idx.pk' % data_name
    
    if not os.path.exists(ref_idx_fname):
        ref_test_idx = np.random.choice(year_idx_dict[max_year], size=test_set_size, replace=False)    
        
        f = open(ref_idx_fname, 'wb')
        pickle.dump(ref_test_idx, f)
        f.close()
    else:
        ref_test_idx = pickle.load(open(ref_idx_fname, 'rb'))
        
    if not os.path.exists(gen_idx_fname):
        gen_test_idx = np.random.choice(range(N), size=test_set_size, replace=False)
        
        f = open(gen_idx_fname, 'wb')
        pickle.dump(gen_test_idx, f)
        f.close()
    else:
        gen_test_idx = pickle.load(open(gen_idx_fname, 'rb'))

    
    ignore_idx = np.concatenate((ref_test_idx,gen_test_idx))
    total_idx_sample_dict = get_avail_idx(train_N+test_set_size, year_idx_dict, ignore_idx, avail_years=[2006, 2007])
    total_idx = get_sample_avail_idx(total_idx_sample_dict)
    train_idx = total_idx[test_set_size:]
    
    # source: take 100 from each train run
    source_test_idx = total_idx[:test_set_size]
    
    X_train = X[train_idx].toarray()
    y_train = y[train_idx]
    
    X_source, y_source = X[source_test_idx].toarray(), y[source_test_idx]
    X_ref, y_ref = X[ref_test_idx].toarray(), y[ref_test_idx]
    X_gen, y_gen = X[gen_test_idx].toarray(), y[gen_test_idx]
    
    if model_name == 'lr':
        model = make_pipeline(StandardScaler(), LogisticRegression())
    elif model_name == 'nn':
        model = make_pipeline(StandardScaler(), MLPClassifier())
    elif model_name == 'svm':
        model = make_pipeline(StandardScaler(), SVC())
    elif model_name == 'xgb':
        model = make_pipeline(StandardScaler(), xgb.XGBClassifier(objective="binary:logistic"))
    
    model.fit(X_train, y_train)

    yhat_source = model.predict(X_source)
    yhat_ref = model.predict(X_ref)
    yhat_gen = model.predict(X_gen)
    
    acc_source = accuracy_score(yhat_source, y_source)
    acc_ref = accuracy_score(yhat_ref, y_ref)
    acc_gen = accuracy_score(yhat_gen, y_gen)
    
    results = {
        'acc_source': acc_source,
        'acc_ref': acc_ref,
        'acc_gen': acc_gen,
        'source_test_idx': source_test_idx,
        'gen_test_idx': gen_test_idx,
        'ref_test_idx': ref_test_idx,
        'yhat_source': yhat_source,
        'yhat_ref': yhat_ref,
        'yhat_gen': yhat_gen,
        'y_source': y_source,
        'y_ref': y_ref,
        'y_gen': y_gen
    }
    
    fname = '%s_%s_%d_sequential.pk' % (data_name, model_name, train_N)
    f = open(fname, 'wb')
    pickle.dump(results, f)
    f.close()
    

    
def run_sequential_wrapper(data, model, n):
    if data == 'mimic':
        X, y, years, groups = get_mimic_sequential_data()

    run_sequential(X, y, years, groups, model, n, data)
    
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n',type=int)
    parser.add_argument('--model')
    parser.add_argument('--data')
    args = parser.parse_args()
    
    run_sequential_wrapper(args.data, args.model, args.n)
