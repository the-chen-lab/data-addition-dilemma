"""
2nd and 3rd section for Yelp

"""

import numpy as np
import pandas as pd
import os.path
import pickle

import scipy
from scipy import sparse
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# turn off xgboost warning about int64
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import xgboost as xgb

DATA_DIR = 'mimic-data/'

tab_cols = ['heartrate_min',
       'heartrate_max', 'heartrate_mean', 'sysbp_min', 'sysbp_max',
       'sysbp_mean', 'diasbp_min', 'diasbp_max', 'diasbp_mean', 'meanbp_min',
       'meanbp_max', 'meanbp_mean', 'resprate_min', 'resprate_max',
       'resprate_mean', 'tempc_min', 'tempc_max', 'tempc_mean', 'spo2_min',
       'spo2_max', 'spo2_mean', 'glucose_min', 'glucose_max', 'glucose_mean',
       'aniongap', 'albumin', 'bicarbonate', 'bilirubin', 'creatinine',
       'chloride', 'glucose', 'hematocrit', 'hemoglobin', 'lactate',
       'magnesium', 'phosphate', 'platelet', 'potassium', 'ptt', 'inr', 'pt',
       'sodium', 'bun', 'wbc', 'had_null', 'oasis', 'oasis_prob', 'sofa',
       'saps', 'sapsii', 'sapsii_prob', 'apsiii', 'apsiii_prob', 'lods']

def run_experiment(n1, n2, n3, model, fname, other_info):
    elect_train, emerg_train, urgen_train, pat_df, source_test, ref_test, gen_test = other_info
    
    n1_idx = np.random.choice(elect_train, n1)
    n2_idx = np.random.choice(emerg_train, n2)
    n3_idx = np.random.choice(urgen_train, n3)
    
    train_idx = np.concatenate([n1_idx, n2_idx, n3_idx])
    training_data = pat_df.iloc[train_idx]
    training_data = get_Xy(training_data)

    return run_model(training_data, source_test, ref_test, gen_test, model, fname)

    
def run_model(training_data, source_test, ref_test, gen_test, model='svm',fname='mimic_results/test.pk'):
    """
    source: source X and y
    ref: reference X and y
    gen: generalizing X and y
    """
    X_train, X2_train, y_train = training_data
    X_s, X2_s, y_s = source_test
    X_r, X2_r, y_r = ref_test
    X_g, X2_g, y_g = gen_test
    
    vec = TfidfVectorizer()
    X_vec_train = vec.fit_transform(X_train)
    X_vec_s = vec.transform(X_s)
    X_vec_r = vec.transform(X_r)
    X_vec_g = vec.transform(X_g)
    
    X_combo_train = hstack([X_vec_train, csr_matrix(X2_train)])
    X_combo_s = hstack([X_vec_s, csr_matrix(X2_s)])
    X_combo_r = hstack([X_vec_r, csr_matrix(X2_r)])
    X_combo_g = hstack([X_vec_g, csr_matrix(X2_g)])
    
    # learn tf idf on the training data, report performance on source, ref, and gen
    
    if model == 'svm' or model == 'lr':
        Cs = [0.01, 0.1, 0.5, 1.0, 1.5, 2.]
    elif model == 'nn':
        Cs = [0.0001, 0.001, 0.01, 0.1]
        
    X_vec_train, X_vec_valid, y_train, y_valid = train_test_split(X_combo_train, y_train, train_size=0.8)
    best_C = None
    best_acc = -float('inf')
    best_clf = None
    for C in Cs:
        if model == 'svm':
            clf = SVC(C=C, probability=True)
        elif model == 'lr':
            clf = LogisticRegression(C=C, penalty='l1',solver='liblinear')
            
        elif model == 'nn':
            clf = MLPClassifier(alpha=C)
            
        clf.fit(X_vec_train,y_train)
        y_hat_valid = clf.predict_proba(X_vec_valid)[:,1]
        iter_acc = accuracy_score(y_valid,y_hat_valid > 0.5)
        
        if iter_acc > best_acc:
            best_acc = iter_acc
            best_C = C
            best_clf = clf
    
    y_s_hat = best_clf.predict_proba(X_combo_s)[:,1]
    y_r_hat = best_clf.predict_proba(X_combo_r)[:,1]
    y_g_hat = best_clf.predict_proba(X_combo_g)[:,1]
    
#     auc_score_ref = roc_auc_score(y_r,y_r_hat)
    acc_score_source = accuracy_score(y_s,y_s_hat > 0.5)
    acc_score_ref = accuracy_score(y_r,y_r_hat > 0.5)
    acc_score_gen = accuracy_score(y_g,y_g_hat > 0.5)
    
    results = {
        'acc_source': acc_score_source,
        'acc_ref': acc_score_ref,
        'acc_gen': acc_score_gen,
        'yhat_s': y_s_hat,
        'yhat_r': y_r_hat,
        'yhat_g': y_g_hat,
        'best_C': best_C,
        'best_acc': best_acc,
        'clf': best_clf,
        'vec': vec
    }
    
    f = open(fname, 'wb')
    pickle.dump(results, f)
    f.close()
    
    return (acc_score_source,acc_score_ref,acc_score_gen)

def get_Xy(df):
    X = df['chartext'].values
    X2 = df[tab_cols].values.astype(float)
    y = df['mort_hosp'].values
    return (X,X2,y)

def subsample(tup,N):
    X, y = tup
    N_total = X.shape[0]
    if N > N_total:
        raise ValueError('Cannot subsample lower than data size')
        
    chosen_idx = np.random.choice(np.arange(N_total), N)
    new_tup = (X[chosen_idx], y[chosen_idx])
    return new_tup

def run_mimic_experiments(n1, n2, n3, model, fdir = 'mimic_results/'):
    pat_df = pd.read_csv(DATA_DIR+'patients_notes_insur.csv')
    tab_df = pd.read_csv(DATA_DIR+'adult_icu_race_mortality.csv')
    
    join_cols = ['subject_id', 'hadm_id', 'icustay_id']
    pat_df = pat_df.merge(tab_df[join_cols+tab_cols], on=join_cols, how='inner')
    pat_df.to_csv('adult_notes_labs_race_mortality.csv')
    
    N_patients, N_feat = pat_df.shape
    
    elect_df = pat_df[pat_df['admType_ELECTIVE'] == 1]
    emerg_df = pat_df[pat_df['admType_EMERGENCY'] == 1]
    urgen_df = pat_df[pat_df['admType_URGENT'] == 1]
    
    elect_idx = np.where(pat_df['admType_ELECTIVE'].values == 1)[0]
    emerg_idx = np.where(pat_df['admType_EMERGENCY'].values == 1)[0]
    urgen_idx = np.where(pat_df['admType_URGENT'].values == 1)[0]
    
    if os.path.exists(fdir+'idx_info.pk'):
        f = open(fdir+'idx_info.pk', 'rb')
        idx_info = pickle.load(f)
        f.close()
        
        elect_test = idx_info['elect_test']
        emerg_test = idx_info['emerg_test']
        urgen_test = idx_info['urgen_test']
        
    else:    
        elect_test = np.random.choice(elect_idx, size=500)
        emerg_test = np.random.choice(emerg_idx, size=500)
        urgen_test = np.random.choice(urgen_idx, size=500)
        
        idx_info = {
        'elect_test': elect_test,
        'emerg_test': emerg_test,
        'urgen_test': urgen_test,
        }

        f = open(fdir+'idx_info.pk', 'wb')
        pickle.dump(idx_info, f)
        f.close()

    
    idx = np.arange(N_patients)
    elect_train = [i for i in elect_idx if i not in elect_test]
    emerg_train = [i for i in emerg_idx if i not in emerg_test]
    urgen_train = [i for i in urgen_idx if i not in urgen_test]
    
    source_test = pat_df.iloc[np.concatenate([elect_test,emerg_test])]
    ref_test = pat_df.iloc[emerg_test]
    gen_test = pat_df.iloc[np.concatenate([elect_test,emerg_test,urgen_test])]
    
    source_test = get_Xy(source_test)
    ref_test = get_Xy(ref_test)
    gen_test = get_Xy(gen_test)
    
    other_info = (elect_train, emerg_train, urgen_train, pat_df, source_test, ref_test, gen_test)
    
    fname = fdir+'results_%s_%d_%d_%d.pk' % (model, n1, n2, n3)
    run_experiment(n1, n2, n3, model, fname, other_info)

def yelp_mix_data_prep(biz_file_source, reviews_file_source, num_sources=5):
    '''
    Takes in yelp data source files and outputs list of csv files of data sources. 
    
    Format of input files from yelp download:
    - yelp_academic_dataset_business.json
    - yelp_academic_dataset_review.json
    
    Number of mixtures is variable, but default is 5 (assume up to 5)
    '''
    
    r_dtypes = {"stars": np.float16, 
            "useful": np.int32, 
            "funny": np.int32,
            "cool": np.int32,
           }

    #open biz files
    with open(biz_file_source, "r") as f:
        biz_df = pd.read_json(f, orient="records", lines=True)
        
    
    #get top n states, to get n sources  
    state_biz_freq = dict(biz_df["state"].value_counts()) #[:5]
    topn_states = dict()
    n = num_sources


    for k in list(state_biz_freq.keys())[:n]:
        topn_states[k] = state_biz_freq[k]
        
    states = list(topn_states.keys())
    
    source_filelog = list()
    for i in range(len(states)):
        
        #get all business ids within state
        print("Adding business ids for %s..."%(states[i]))
        N1_A = biz_df[biz_df["state"]==states[i]]
        N1_A_ids = set(N1_A['business_id'].values)

        

        #get all reviews associated with restaurants in state 
        N1_A_reviews = list()
        print("Adding reviews for %s..."%(states[i]))
        with open(reviews_file_source, "r") as f:
            reader = pd.read_json(f, orient="records", lines=True, 
                          dtype=r_dtypes, chunksize=1000)
        
            for chunk in reader:
                reduced_chunk = chunk.drop(columns=['review_id', 'user_id'])\
                             .query("`business_id` in @N1_A_ids")
                N1_A_reviews.append(reduced_chunk)
                #break


            N1_A_reviews = pd.concat(N1_A_reviews, ignore_index=True)

        print("Generating source files...")
        N1_A_final = pd.merge(N1_A_reviews, N1_A, on="business_id", how="inner")
        csv_filename = "N_%d_%s_final.csv"%(i, states[i]) 
        N1_A_final.to_csv(csv_filename)
        
        source_filelog.append(csv_filename)
        
    return source_filelog
    


def get_yelp_mixture_data(num_sources=5):
    """
    X: features 
    y: outcomes
    source: lst of source labels (length N)
    groups: lst of sensitive attributes (length N)
    """
    
    sources = 100
    # TODO
    min_yr = 2006
    max_yr = 2009
    source_dir = './'
    return X, y, sources, groups


def get_mimic_mixture_data():
    df = pd.read_csv('adult_notes_labs_race_mortality.csv')
    
    chartext = df['chartext'].apply(lambda x: x[:1000])
    chartext = chartext.values
    
    vec = TfidfVectorizer()
    X_text_vec = vec.fit_transform(chartext).tocsr()
    X_tab = df[tab_cols].values
    X_tab = X_tab.astype(float)
    
    X = sparse.hstack((X_text_vec, X_tab)).tocsr()
    y = (df['mort_hosp'].values).astype(int)
    
    adm_cols = [i for i in df.columns if 'admType' in i]
    eth_cols = [i for i in df.columns if 'eth_' in i]

    df['admType'] = df[adm_cols].idxmax(axis=1)
    df['eth'] = df[eth_cols].idxmax(axis=1)

    sources = df['admType'].values
    groups = df['eth'].values
    
    return X, y, sources, groups

def is_source_i(sources, i):
    # s: lst of sources (length N)
    # i: idx of source we are looking for
    return str(i) in s

def run_mixture(X, y, sources, groups, model_name, sources_n, data_name, test_set_size=1000):
    """
    X: features
    y: outcomes
    sources: lst of sources (alpha
    groups: lst of sensitive attributes
    
    model_name: lr, nn, svm, or xgb
    data_name: yelp, mimic, or folk
    sources_n: 
    """
    
    # step 1: collect training set from random samples from each source at given source_n
    # step 2a: gen_test is random sample from all three (e.g., 1000) - FIXED
    # step 2b: ref_test is random sample from one of them (e.g., source1) - FIXED
    # step 2c: source_test is random sample from the training data of that run
    #  -> sources_n / N -> ratio needed to put in source_test (* 1000)
    # step 3: train and save output
    
    # decision: tfidf earlier; don't use for train
    N = X.shape[0]
    # sum of sources_n
    N_sources = np.sum(sources_n)
    
    ref_idx_fname = '%s_ref_idx_mixture.pk' % data_name
    gen_idx_fname = '%s_gen_idx_mixture.pk' % data_name
    sources_uniq = np.unique(sources)
    ref_source_name = sources_uniq[0]
    
    ref_source_idx_all = np.where(sources == ref_source_name)[0]
    
    # ref set: sampled from only one source
    if not os.path.exists(ref_idx_fname):
        ref_test_idx = np.random.choice(ref_source_idx_all, size=test_set_size, replace=False)    
        
        f = open(ref_idx_fname, 'wb')
        pickle.dump(ref_test_idx, f)
        f.close()
    else:
        ref_test_idx = pickle.load(open(ref_idx_fname, 'rb'))
    
    # generative set: sampled from all sources
    if not os.path.exists(gen_idx_fname):
        gen_test_idx = np.random.choice(range(N), size=test_set_size, replace=False)
        
        f = open(gen_idx_fname, 'wb')
        pickle.dump(gen_test_idx, f)
        f.close()
    else:
        gen_test_idx = pickle.load(open(gen_idx_fname, 'rb'))
    
    ignore_idx = set(np.concatenate((ref_test_idx,gen_test_idx)))
    
    all_idx = range(N)
    # avail_idx = [i for i in all_idx if i not in ignore_idx]
    # avail_train_idx, source_test_idx = train_test_split(avail_idx, test_size=test_set_size,shuffle=True)
    train_idx = list()
    source_test_idx = list()
    
    for source_i, num_source_train_needed in enumerate(sources_n):
        source_i_idx = np.where(sources == sources_uniq[source_i])[0]
        source_i_idx = np.array([i for i in source_i_idx if i not in ignore_idx])
        
        num_source_test_needed = int(num_source_train_needed / N_sources * test_set_size)
        try:
            source_i_test_train_idx = np.random.choice(source_i_idx, size=(num_source_test_needed+num_source_train_needed), replace=False)
        except:
            source_i_test_train_idx = np.random.choice(source_i_idx, size=(num_source_test_needed+num_source_train_needed), replace=True)
        
        source_i_test = source_i_test_train_idx[:num_source_test_needed]
        source_i_train = source_i_test_train_idx[num_source_test_needed:]
        
        train_idx.append(source_i_train)
        source_test_idx.append(source_i_test)
    
    train_idx = np.concatenate(train_idx)
    source_test_idx = np.concatenate(source_test_idx)
    
    # only tabular
    # if not scipy.sparse.issparse(X):
    X_train = X[train_idx]
    X_source, y_source = X[source_test_idx], y[source_test_idx]
    X_ref, y_ref = X[ref_test_idx], y[ref_test_idx]
    X_gen, y_gen = X[gen_test_idx], y[gen_test_idx]

    # update: not using standard scaler anymore
    # text + tabular data
    # else:
    #     X_train = X[train_idx].toarray()
    #     X_source, y_source = X[source_test_idx].toarray(), y[source_test_idx]
    #     X_ref, y_ref = X[ref_test_idx].toarray(), y[ref_test_idx]
    #     X_gen, y_gen = X[gen_test_idx].toarray(), y[gen_test_idx]
        
    y_train = y[train_idx]
    
    # import pdb; pdb.set_trace()
    if model_name == 'lr':
        model = LogisticRegression()
    elif model_name == 'nn':
        model = MLPClassifier()
    elif model_name == 'svm':
        model = SVC()
    elif model_name == 'xgb':
        model = xgb.XGBClassifier(objective="binary:logistic")
    
    model.fit(X_train, y_train)

    yhat_train = model.predict(X_train)
    yhat_source = model.predict(X_source)
    yhat_ref = model.predict(X_ref)
    yhat_gen = model.predict(X_gen)
    
    
    acc_train = accuracy_score(yhat_train, y_train)
    acc_source = accuracy_score(yhat_source, y_source)
    acc_ref = accuracy_score(yhat_ref, y_ref)
    acc_gen = accuracy_score(yhat_gen, y_gen)
    
    
    results = {
        'acc_source': acc_source,
        'acc_ref': acc_ref,
        'acc_gen': acc_gen,
        'acc_train': acc_train,
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
    
    # train_N = np.sum(sources_n)
    n1 = sources_n[0]
    n2 = sources_n[1]
    n3 = sources_n[2]
    fname = 'results/%s_%s_%d_%d_%d_mixture.pk' % (data_name, model_name, n1, n2, n3)
    f = open(fname, 'wb')
    pickle.dump(results, f)
    f.close()
    
#     # learn tf idf on the training data, report performance on source, ref, and gen
    
#     if model == 'svm' or model == 'lr':
#         Cs = [0.01, 0.1, 0.5, 1.0, 1.5, 2.]
#     elif model == 'nn':
#         Cs = [0.0001, 0.001, 0.01, 0.1]
        
#     X_vec_train, X_vec_valid, y_train, y_valid = train_test_split(X_combo_train, y_train, train_size=0.8)
#     best_C = None
#     best_acc = -float('inf')
#     best_clf = None
#     for C in Cs:
#         if model == 'svm':
#             clf = SVC(C=C, probability=True)
#         elif model == 'lr':
#             clf = LogisticRegression(C=C, penalty='l1',solver='liblinear')
            
#         elif model == 'nn':
#             clf = MLPClassifier(alpha=C)
            
#         clf.fit(X_vec_train,y_train)
#         y_hat_valid = clf.predict_proba(X_vec_valid)[:,1]
#         iter_acc = accuracy_score(y_valid,y_hat_valid > 0.5)
        
#         if iter_acc > best_acc:
#             best_acc = iter_acc
#             best_C = C
#             best_clf = clf
    
#     y_s_hat = best_clf.predict_proba(X_combo_s)[:,1]
#     y_r_hat = best_clf.predict_proba(X_combo_r)[:,1]
#     y_g_hat = best_clf.predict_proba(X_combo_g)[:,1]
    
# #     auc_score_ref = roc_auc_score(y_r,y_r_hat)
#     acc_score_source = accuracy_score(y_s,y_s_hat > 0.5)
#     acc_score_ref = accuracy_score(y_r,y_r_hat > 0.5)
#     acc_score_gen = accuracy_score(y_g,y_g_hat > 0.5)
    
#     results = {
#         'acc_source': acc_score_source,
#         'acc_ref': acc_score_ref,
#         'acc_gen': acc_score_gen,
#         'yhat_s': y_s_hat,
#         'yhat_r': y_r_hat,
#         'yhat_g': y_g_hat,
#         'best_C': best_C,
#         'best_acc': best_acc,
#         'clf': best_clf,
#         'vec': vec
#         # TODO: train_accuracy
#     }
    
#     f = open(fname, 'wb')
#     pickle.dump(results, f)
#     f.close()
    return



def run_mixture_wrapper(n1, n2, n3, n4, n5, model, data):
    if data == 'mimic':
        X, y, sources, groups = get_mimic_mixture_data()
    elif data == 'yelp':
        X, y, sources, groups = get_yelp_mixture_data()
        
    # check if results/ folder exists; if not, create it
    if not os.path.exists('results/'):
        os.makedirs('results/')
        print('Created results/ directory')
        
    
    sources_n = []
    for n in [n1, n2, n3, n4, n5]:
        if n is not None:    
            sources_n.append(n)
    if data == 'mimic':
        len_sources_n = len(sources_n)
        if len_sources_n > 3:
            sources_n = sources_n[:3]
        
    run_mixture(X, y, sources, groups, model, sources_n, data)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n1',type=int, default=0)
    parser.add_argument('--n2',type=int, default=0)
    parser.add_argument('--n3',type=int, default=0)
    parser.add_argument('--n4',type=int, default=0)
    parser.add_argument('--n5',type=int, default=0)
    parser.add_argument('--model')
    parser.add_argument('--data')
    args = parser.parse_args()
    
    run_mixture_wrapper(args.n1, args.n2, args.n3, args.n4, args.n5, args.model, args.data)

    