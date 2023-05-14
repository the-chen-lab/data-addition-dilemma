# similar to run_mimic
# python run_sequential.py --data mimic --model lr -n 1000 (and etc)

import numpy as np
import os
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

def run_yelp_exp_prep(source_dir, min_yr, max_yr):
    '''
    Takes in directory of data sources csv files and outputs formatted X, Y for model training.
    '''
    
    df = pd.DataFrame()

    for yr in yr_range:
        #/Users/rajiinio/Documents/more-data-more-problems/mdmp_data_clean/2005_2004_final_dd.csv
        source_file = "/%d_%d_final_dd.csv" %(yr+1, yr)
        source_path = source_dir+source_file
        df_yr = pd.read_csv(source_path)
    
        #clean up
        df_yr["year"] = yr
    
        #TO DO: properly clean categories 
        #df_yr['categories_lst'] = df_yr['categories'].apply(lambda x: x.split(', '))
        df = df.append(df_yr)
        #df = pd.concat([df, df_yr])
    
    
    tab_cols = ['stars_x','useful', 'funny', 'cool']

    vec = TfidfVectorizer()
    X_text_vec = vec.fit_transform(df['text']).tocsr()
    X_tab = df[tab_cols].values
    X_tab = X_tab.astype(float)
    #X = sparse.hstack((X_text_vec, X_tab)).tocsr()
    X = sparse.hstack((X_text_vec, X_tab)).tocsr()

    y = (df['stars_y'].values >= 3.).astype(int)
    
    
    return X, y

def yelp_seq_data_prep(biz_file_source, reviews_file_source, max_yr, min_yr):
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

    
    with open(biz_file_source, "r") as f:
        biz_df = pd.read_json(f, orient="records", lines=True)
        
    source_filelog = list()
    for i in range(min_yr, max_yr+1):
        reviews_df = []

    
        with open(reviews_file_source, "r") as f:
            reader = pd.read_json(f, orient="records", lines=True, 
                          dtype=r_dtypes, chunksize=1000)
            
            print("Adding reviews for %d..."%(i))
            for chunk in reader:
                date_query = "'%d-01-01'> `date` >= '%d-01-01'"%(i+1, i)
                #print(date_query)
                reduced_chunk = chunk.drop(columns=['review_id', 'user_id'])\
                             .query(date_query) 
                reviews_df.append(reduced_chunk)
                #break
    
            reviews_df = pd.concat(reviews_df, ignore_index=True)


        print("Generating source files...")
        reviews_df_final = pd.merge(reviews_df, biz_df, on="business_id", how="inner") 
        csv_filename = "%d_%d_yelp.csv"%(i+1, i)
        reviews_df_final.to_csv(csv_filename)
        
        source_filelog.append(csv_filename)
        
    return source_filelog


def get_mimic_sequential_data():
    """
    output
     - X: numpy array or csr matrix
     - y: np array of outcomes
     - years:
     - groups: 
    """
    df = pd.read_csv('adult_icu_year.csv')
    tab_cols = [
    'age', 'first_hosp_stay', 'first_icu_stay', 'eth_asian', 
    'eth_black', 'eth_hispanic', 'eth_other', 'eth_white', 'heartrate_min', 
    'heartrate_max', 'heartrate_mean', 'sysbp_min', 'sysbp_max', 
    'sysbp_mean', 'diasbp_min', 'diasbp_max', 'diasbp_mean', 'meanbp_min', 
    'meanbp_max', 'meanbp_mean', 'resprate_min', 'resprate_max', 
    'resprate_mean', 'tempc_min', 'tempc_max', 'tempc_mean', 'spo2_min', 
    'spo2_max', 'spo2_mean', 'glucose_min', 'glucose_max', 'glucose_mean', 
    'aniongap', 'albumin', 'bicarbonate', 'bilirubin', 'creatinine', 
    'chloride', 'glucose', 'hematocrit', 'hemoglobin', 'lactate', 'magnesium', 
    'phosphate', 'platelet', 'potassium', 'ptt', 'inr', 'pt', 'sodium', 
    'bun', 'wbc'
    ]
    
    # X.shape = (35131, 52)
    X = df[tab_cols].values
    y = df['mort_hosp'].values
    years = df['anchor_year_group'].values
    groups = df['insurance'].values
    return X, y, years, groups
    
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
    
    def valid_year(year):
        if avail_years is None:
            return True
        else:
            return year in avail_years
            
    for year in sorted(year_idx_dict_ignore.keys()):
        if valid_year(year) and remaining > 0:
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

def run_sequential(X,y,years,groups,model_name,train_N, data_name, test_set_size=1000):
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
    total_idx_sample_dict = get_avail_idx(train_N+test_set_size, year_idx_dict, ignore_idx, avail_years=None)
    total_idx = get_sample_avail_idx(total_idx_sample_dict)
    train_idx = total_idx[test_set_size:]
    
    # source: take 100 from each train run
    source_test_idx = total_idx[:test_set_size]
    
    try:
        X_train = X[train_idx]
    except:
        X_train = X[train_idx].toarray()
    y_train = y[train_idx]
    
    try:
        X_source, y_source = X[source_test_idx], y[source_test_idx]
        X_ref, y_ref = X[ref_test_idx], y[ref_test_idx]
        X_gen, y_gen = X[gen_test_idx], y[gen_test_idx]
    except:
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
    
    fname = 'results/%s_%s_%d_sequential.pk' % (data_name, model_name, train_N)
    f = open(fname, 'wb')
    pickle.dump(results, f)
    f.close()
    

    
def run_sequential_wrapper(data, model, n):
    if data == 'mimic':
        X, y, years, groups = get_mimic_sequential_data()
    elif data == 'yelp':
        X, y, years, groups = get_yelp_sequential_data()
        
    # check if results/ folder exists; if not, create it
    if not os.path.exists('results/'):
        os.makedirs('results/')
        print('Created results/ directory')
        
    run_sequential(X, y, years, groups, model, n, data)
    
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-n',type=int)
    parser.add_argument('--model')
    parser.add_argument('--data')
    args = parser.parse_args()
    
    run_sequential_wrapper(args.data, args.model, args.n)
