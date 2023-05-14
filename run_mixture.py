import numpy as np
import pandas as pd
import os.path
import pickle

from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

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
    


#def run_yelp_exp():


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n1',type=int)
    parser.add_argument('--n2',type=int)
    parser.add_argument('--n3',type=int)
    parser.add_argument('--model')
    args = parser.parse_args()
    
    run_mimic_experiments(args.n1, args.n2, args.n3, args.model)
