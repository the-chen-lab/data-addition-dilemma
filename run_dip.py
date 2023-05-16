
import numpy as np
from scipy.sparse import vstack
import itertools
import seaborn as sns
from scipy import sparse
import os
import pandas as pd
import pickle
import random

from xgboost import XGBClassifier
from multiprocessing import Pool

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import GradientBoostingClassifier


from folktables import ACSDataSource, ACSEmployment, ACSIncome

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(
    # font='Franklin Gothic Book',
rc={
 'axes.axisbelow': False,
 'axes.edgecolor': 'lightgrey',
 'axes.facecolor': 'None',
 'axes.grid': False,
 'axes.labelcolor': 'dimgrey',
 'axes.spines.right': False,
 'axes.spines.top': False,
 'figure.facecolor': 'white',
 'lines.solid_capstyle': 'round',
 'patch.edgecolor': 'w',
 'patch.force_edgecolor': True,
 'text.color': 'dimgrey',
 'xtick.bottom': False,
 'xtick.color': 'dimgrey',
 'xtick.direction': 'out',
 'xtick.top': False,
 'ytick.color': 'dimgrey',
 'ytick.direction': 'out',
 'ytick.left': False,
 'ytick.right': False})
sns.set_context("notebook", rc={"font.size":14,
"axes.titlesize":14,
"axes.labelsize":14})

from matplotlib.ticker import FuncFormatter, FormatStrFormatter

import warnings
warnings.filterwarnings('ignore')

def flatten(lst):
    return list(itertools.chain(*lst))

def get_clean_race(s0):
        s = s0.lower()

        check_races = ['white', 'black', 'asian', 'hispanic', 'unknown']
        for r in check_races:
            if r in s:
                return r
        return 'other'
    
def get_group(s):
    asian_cats = ['Japanese',
                'Chinese',
                'Southern',
                'Vietnamese',
                'Asian Fusion','Thai']
    ethnic_categories = [
                'American',
                'Italian',
                'Mexican',
                'Japanese',
                'Chinese',
                'Southern',
                'Vietnamese',
                'Asian Fusion',
                'Mediterranean',
                'Thai'
                ]
    
    for cat in ethnic_categories:
        
        try:
            if cat in s:
                if cat in asian_cats:
                    return 'Asian'
                else:
                    return cat
        except:
            return 'Other'
    else:
        return 'Other'

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

def run_yelp_exp_prep(source_dir, min_yr, max_yr,cache=True):
    '''
    Takes in directory of data sources csv files and outputs formatted X, Y for model training.
    '''
    if not cache:
        print('Not using cache, creating Yelp pk file...')
        df_lst = []
        yr_range = range(min_yr, max_yr+1)

        for yr in yr_range:
            #/Users/rajiinio/Documents/more-data-more-problems/mdmp_data_clean/2005_2004_final_dd.csv
            source_file = "/%d_%d_yelp.csv" %(yr+1, yr)
            source_path = source_dir+source_file
            df_yr = pd.read_csv(source_path)

            #clean up
            df_yr["year"] = yr
            df_yr['group'] = df_yr['categories'].apply(get_group)
            df_lst.append(df_yr)
            #TO DO: properly clean categories 
            # df_yr['categories_lst'] = df_yr['categories'].apply(lambda x: x.split(', '))
        df = pd.concat(df_lst)

        tab_cols = ['stars_x','useful', 'funny', 'cool']
    
        text = df['text'].apply(lambda x: x[:1000]).values
        
        vec = TfidfVectorizer()
        X_text_vec = vec.fit_transform(text).tocsr()
        X_tab = df[tab_cols].values
        X_tab = X_tab.astype(float)
        #X = sparse.hstack((X_text_vec, X_tab)).tocsr()
        X = sparse.hstack((X_text_vec, X_tab)).tocsr()

        y = (df['stars_y'].values).astype(int)
        
        f = open('yelp_seq.pk','wb')
        years = df['year'].values
        groups = pd.Categorical(df['group'], categories=df['group'].unique()).codes
        states = df['state'].values
        
        pickle.dump((X, y, years, groups, states), f)
        f.close()
    else:
        f = open('yelp_seq.pk','rb')
        X, y, years, groups, states = pickle.load(f)
        f.close()
    return X, y, years, groups, states

def get_yelp_data():
    print('loading Yelp data...')
    min_yr = 2006
    max_yr = 2009
    source_dir = './'
    
    biz_file_source = '../../yelp-data/yelp_academic_dataset_business.json'
    reviews_file_source = '../../yelp-data/yelp_academic_dataset_review.json'
    min_yr, max_yr = 2006, 2010
    source_dir = '../'
    
    # see if we can find the files
    if not os.path.exists(source_dir+'%d_%d_yelp.csv' % (min_yr+1, min_yr)):
        print('Creating yelp data csvs...')
        yelp_seq_data_prep(biz_file_source, reviews_file_source, max_yr, min_yr)
        
    X,y,years,groups,states = run_yelp_exp_prep(source_dir, min_yr, max_yr, cache=True)
    
    return X, y, groups, states, years


def get_folktables_data(cache=True):
    print('loading folktables data...')
    # state='SD'
    if not cache:
        all_features = list()
        all_labels = list()
        all_groups = list()
        all_states = list()
        all_years = list()

        for state in ['CA', 'HI', 'SD', 'PA', 'MI', 'GA', 'MS']:
            # data_dict[state] = {}
            for year in [2014, 2015, 2016, 2017, 2018]: 
                data_source = ACSDataSource(survey_year=str(year), horizon='1-Year', survey='person')
                acs_data = data_source.get_data(states=[state], download=True)
                # data_dict[state][year] = {}
                # data_dict[state][year]['x'] = features

                features, labels, groups = ACSIncome.df_to_numpy(acs_data)

                all_features.append(features)
                all_labels.append(labels)
                all_groups.append(groups)


                N = features.shape[0]
                year_lst = np.array([year] * N)
                state_lst = np.array([state] * N)

                all_states.append(state_lst)
                all_years.append(year_lst)

                # data_dict[state][year]['y'] = label
                # data_dict[state][year]['g'] = group 

        X = np.concatenate(all_features)
        y = np.concatenate(all_labels)
        groups = np.concatenate(all_groups)
        states = np.concatenate(all_states)
        years = np.concatenate(all_years)
        
        f = open('folktables.pk','wb')
        pickle.dump((X,y,groups,states,years), f)
        f.close()
        
    else:
        (X,y,groups,states,years) = pickle.load(open('folktables.pk', 'rb'))
    
    return X, y, groups, states, years
    
    
def get_mimic_data(LABEL1, LABEL2, LABEL_FIELD):    
    print('loading MIMIC data...')
    df = pd.read_csv('../mimic-data/mimic_diagnoses.csv')
    
    df['diagnoses'] = df['diagnoses'].apply(lambda x: x.replace(' <sep> ', ' '))
    df['procedure'] = df['procedure'].apply(lambda x: x.replace(' <sep> ', ' '))
    df['diag_proc'] = df.apply(lambda x: x['diagnoses'] + x['procedure'], axis=1)
    

    df['race'] = df['race'].apply(get_clean_race)
    
    df['race_full'] = df['race'].copy()
    uniq_race = df['race_full'].unique()
    n_groups = len(uniq_race)
    df['race'] = pd.Categorical(df['race_full'], categories=uniq_race).codes

    print('feature engineering...')
    diagproc = df['diag_proc'].values

    vec = CountVectorizer(binary=True)
    X = vec.fit_transform(diagproc).tocsr()
    y = df['readmission'].values
    groups = df['race'].values
    states = df[LABEL_FIELD].values
    years = df['real_admit_year'].values
    return X, y, groups, states, years

def part1_worker(X, y, groups, states, years, start_year, run, clf, clf_dict):
    """
    function for Pool
    """
    results = list()
    year = start_year
    year_idx = np.where(years == year)[0]
    X_year = X[year_idx]
    y_year = y[year_idx]
    g_year = groups[year_idx]
    n_groups = len(np.unique(groups))
    
    X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
                X_year,
                y_year, 
                g_year, test_size=0.2)

    model = make_pipeline(clf_dict[clf]())

    model.fit(X_train, y_train)

    yhat = model.predict(X_test)

    group_acc_lst = list()
    for group in range(n_groups):
        group_acc = accuracy_score(y_test[(group_test == group)], yhat[(group_test == group)])
        group_acc_lst.append(group_acc)
    group_acc_lst = np.array(group_acc_lst)

    results.append({
        'year': year, 
        'test_acc': model.score(X_test, y_test), 
        'worst': np.nanmin(group_acc_lst),
        'EO': np.nanmax(group_acc_lst) - np.nanmin(group_acc_lst), 
        'size': len(y_train), 
        'run': run, 
        'clf': clf
    })

    for year in sorted(np.unique(years)): 
        year_idx = np.where(years == year)[0]
        X_year = X[year_idx]
        y_year = y[year_idx]
        g_year = groups[year_idx]

        X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
                    X_year,
                    y_year, 
                    g_year, test_size=0.4)

        yhat = model.predict(X_test)
        
        group_acc_lst = list()
        for group in range(n_groups):
            group_acc = accuracy_score(y_test[(group_test == group)], yhat[(group_test == group)])
            group_acc_lst.append(group_acc)
        group_acc_lst = np.array(group_acc_lst)

        # import pdb; pdb.set_trace()
        results.append({
        'year': year, 
        'test_acc': model.score(X_test, y_test), 
        'worst': np.nanmin(group_acc_lst),
        'EO': np.nanmax(group_acc_lst) - np.nanmin(group_acc_lst), 
        'size': len(y_train), 
        'run': run, 
        'clf': clf,
        # 'group_acc_lst': group_acc_lst
    })
    return results


def part1(X, y, groups, states, years, clf_dict, start_year=2008, num_trials=5, fname='figures/years.pdf'):
    #### 1. Is there distribution shift across years?
    print('part 1...')
    
    # debug
    # results = (X, y, groups, states, years, start_year, 0, 'LR', clf_dict) 
    # results = part1_worker(X, y, groups, states, years, start_year, 0, 'LR', clf_dict)
    with Pool(processes=15) as pool:
        args = [(X, y, groups, states, years, start_year, run, clf, clf_dict) for run in range(num_trials) for clf in clf_dict]
        results = pool.starmap(part1_worker, args)
    
    results_df = pd.DataFrame(flatten(results))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    sns.lineplot(data=results_df, x='year', y='test_acc', hue='clf', ax=axes[0])
    axes[0].set_title("Accuracy of %d model on all years" % start_year,fontsize=12)
    axes[0].xaxis.set_major_formatter(FormatStrFormatter('%d'))
    axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # axes[0].xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1}'.format(y)))
    
    sns.lineplot(data=results_df, x='year', y='EO', hue='clf', ax=axes[1])
    axes[1].set_title("EO of %d model on all years" % start_year,fontsize=12)
    axes[1].xaxis.set_major_formatter(FormatStrFormatter('%d'))
    axes[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    sns.lineplot(data=results_df, x='year', y='worst', hue='clf', ax=axes[2])
    axes[2].set_title("Worst group perf of %d model" % start_year,fontsize=12)
    axes[2].xaxis.set_major_formatter(FormatStrFormatter('%d'))
    axes[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    plt.savefig(fname, bbox_inches='tight')
    
    csv_fname = fname.replace('.pdf','.csv').replace('figures/','csv/')
    results_df.to_csv(csv_fname)
        
def part2_worker(X, y, groups, states, years, run, clf, clf_dict, LABEL1):
    results = []
    ref_state = LABEL1
    state_idx = np.where(states == ref_state)[0]
    
    X_state = X[state_idx]
    y_state = y[state_idx]
    g_state = groups[state_idx]
    n_groups = len(np.unique(groups))
    state_count_dict = pd.Series(states).value_counts().to_dict()
    uniq_states = [i for i in np.unique(states) if state_count_dict[i] > 50]

    X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
                X_state,
                y_state, 
                g_state, test_size=0.2)

    model = make_pipeline(clf_dict[clf]())

    model.fit(X_train, y_train)


    yhat = model.predict(X_test)

    group_acc_lst = list()
    for group in range(n_groups):
        group_acc = accuracy_score(y_test[(group_test == group)], yhat[(group_test == group)])
        group_acc_lst.append(group_acc)
    group_acc_lst = np.array(group_acc_lst)

    results.append({
        'state': ref_state, 
        'test_acc': model.score(X_test, y_test), 
        'worst':np.nanmin(group_acc_lst),
        'EO': np.nanmax(group_acc_lst) - np.nanmin(group_acc_lst), 
        'size': len(y_train), 
        'run': run, 
        'clf': clf,
    })
    
    for state in uniq_states:
        if state != ref_state: 
            state_idx = np.where(states == state)[0]

            # year_idx = np.where(years == year)[0]
            X_state = X[state_idx]
            y_state = y[state_idx]
            g_state = groups[state_idx]
            
            X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
                    X_state,
                    y_state, 
                    g_state, test_size=0.4)

            yhat = model.predict(X_test)
            group_acc_lst = list()
            for group in range(n_groups):
                group_acc = accuracy_score(y_test[(group_test == group)], yhat[(group_test == group)])
                group_acc_lst.append(group_acc)
            group_acc_lst = np.array(group_acc_lst)

            results.append({
            'state': state, 
            'test_acc': model.score(X_test, y_test), 
            'EO': np.nanmax(group_acc_lst) - np.nanmin(group_acc_lst), 
            'worst':np.nanmin(group_acc_lst),
            'size': len(y_train), 
            'run': run, 
            'clf': clf,
        })
    return results

def part2(X, y, groups, states, years, clf_dict, LABEL1, num_trials=5, fname='figures/mimic_p2_states.pdf'):
    print('part 2...')
    
    # part2_worker(X, y, groups, states, years, 0, 'LR', clf_dict, LABEL1)
    with Pool(processes=15) as pool:
        args = [(X, y, groups, states, years, run, clf, clf_dict, LABEL1) for run in range(num_trials) for clf in clf_dict]
        results = pool.starmap(part2_worker, args)

    results_df = pd.DataFrame(flatten(results))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    
    sns.barplot(data=results_df, x='state', y='test_acc', hue='clf', ax=axes[0])
    axes[0].set_title(f"Accuracy of {LABEL1} model on other groups",fontsize=12)
    sns.barplot(data=results_df, x='state', y='EO', hue='clf', ax=axes[1])
    axes[1].set_title(f"EO of {LABEL1} model on other groups",fontsize=12)
    sns.barplot(data=results_df, x='state', y='worst', hue='clf', ax=axes[2])
    axes[2].set_title(f"Worst group perf of {LABEL1} model",fontsize=12)

    # axes[0].set_xticks(rotation=90)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation = 45, ha='right')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation = 45, ha='right')
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation = 45, ha='right')
    plt.savefig(fname, bbox_inches='tight')
    
    csv_fname = fname.replace('.pdf','.csv').replace('figures/','csv/')
    results_df.to_csv(csv_fname)
    
def part3_worker(X, y, groups, states, years, clf, clf_dict, state, run, size):
    results = []
    n_groups = len(np.unique(groups))
    
    state_idx = np.where(states == state)[0]

    X_state = X[state_idx]
    y_state = y[state_idx]
    g_state = groups[state_idx]

    X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
        X_state, 
        y_state, 
        g_state, test_size=0.2, random_state=run)

    # set size of training set
    incl = np.asarray(random.sample(range(len(y_train)), size))

    X_train = X_train[incl]
    y_train = y_train[incl]
    model = make_pipeline(clf_dict[clf]())

    model.fit(X_train, y_train)

    yhat = model.predict(X_test)

    group_acc_lst = list()
    for group in range(n_groups):
        group_acc = accuracy_score(y_test[(group_test == group)], yhat[(group_test == group)])
        group_acc_lst.append(group_acc)
    group_acc_lst = np.array(group_acc_lst)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    results.append({
        'train_acc': train_acc, 
        'test_acc': test_acc, 
        'EO': np.nanmax(group_acc_lst) - np.nanmin(group_acc_lst), 
        'worst': np.nanmin(group_acc_lst),
        'size': len(y_train), 
        'run': run, 
        'clf': clf, 
    })
    return results

def part3(X, y, groups, states, years, clf_dict, LABEL1, LABEL2, num_trials=5, fname='figures/mimic_p3_moredata.pdf'):
    print('part 3...')
        
    results = [] 
    # state='EU OBSERVATION'
    # state='DIRECT EMER.'
    state=LABEL2
    state_idx = np.where(states == state)[0]

    if 'folktables' in fname:
        size_arr = [50, 100, 500, 1000, 2000, 4000, 8000, 12000, 14000]
    else:
        start = np.log10(500)
        stop = np.log10(len(state_idx)*0.8)
        size_arr = [int(i) for i in np.logspace(start,stop,10)]
    
    
    with Pool(processes=15) as pool:
        args = [(X, y, groups, states, years, clf, clf_dict, state, run, size) for run in range(num_trials) for clf in clf_dict for size in size_arr]
        results = pool.starmap(part3_worker, args)
                
    results_df = pd.DataFrame(flatten(results))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    sns.lineplot(data=results_df, x='size', y='test_acc', hue='clf', ax=axes[0])
    axes[0].set_title("Accuracy with increasing training datasize",fontsize=14)
    axes[0].set_xscale('log')
    sns.lineplot(data=results_df, x='size', y='EO', hue='clf', ax=axes[1])
    axes[1].set_title("EO with increasing training datasize",fontsize=14)
    axes[1].set_xscale('log')
    
    sns.lineplot(data=results_df, x='size', y='worst', hue='clf', ax=axes[2])
    axes[2].set_title("Worst group perf with\nincreasing training datasize",fontsize=14)
    axes[2].set_xscale('log')
    plt.savefig(fname,bbox_inches='tight')
    
    csv_fname = fname.replace('.pdf','.csv').replace('figures/','csv/')
    results_df.to_csv(csv_fname)

def part4_worker(X, y, groups, states, years, clf, clf_dict, state, run, size,X_state2, y_state2, g_state2):
    """
    state: smaller and less acc state 
    """
    results = []
    
    state_idx = np.where(states == state)[0]

    X_state = X[state_idx]
    y_state = y[state_idx]
    g_state = groups[state_idx]
    
    n_groups = len(np.unique(groups))

    # if size < X_state.shape[0]:
    X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
        X_state, 
        y_state, 
        g_state, test_size=0.2, random_state=run)

    if size < X_train.shape[0]:                
        incl = np.asarray(random.sample(range(len(y_train)), size))
        X_train_small = X_train[incl]
        y_train_small = y_train[incl]
    else:
        idx = np.arange(X_state.shape[0])
        np.random.shuffle(idx)

        X_joint = vstack((X_train,X_state2[idx]))
        y_joint = np.concatenate((y_train,y_state2[idx]))

        X_train_small, y_train_small = X_joint[:size], y_joint[:size]

    model = make_pipeline(clf_dict[clf]())
    # model.fit(X_joint[:size], y_joint[:size])
    model.fit(X_train_small, y_train_small)

    yhat = model.predict(X_test)

    group_acc_lst = list()
    for group in range(n_groups):
        group_acc = accuracy_score(y_test[(group_test == group)], yhat[(group_test == group)])
        group_acc_lst.append(group_acc)
    group_acc_lst = np.array(group_acc_lst)

    train_acc = model.score(X_train_small, y_train_small)
    test_acc = model.score(X_test, y_test)

    results.append({
        'train_acc': train_acc, 
        'test_acc': test_acc, 
        'worst': np.nanmin(group_acc_lst),
        'EO': np.nanmax(group_acc_lst) - np.nanmin(group_acc_lst), 
        'size': size, 
        'run': run, 
        'clf': clf, 
    })
    return results

def part4(X, y, groups, states, years, clf_dict, LABEL1, LABEL2, num_trials=5, fname='figures/mimic_p4_dip.pdf'):
    print('part 4...')
 
    state = LABEL2
    state2 = LABEL1

    state_idx = np.where(states == state)[0]
    state2_idx = np.where(states == state2)[0]
    
    if 'folktables' in fname:
        size_arr = [50, 100, 500, 1000, 2000, 4000, 8000]
    else:
        size_arr1 = np.logspace(np.log10(200),np.log10(len(state_idx) * 0.8), 10)
        size_arr2 = np.logspace(np.log10(max(size_arr1)),np.log10(max(size_arr1) + len(state2_idx)), 5)

        size_arr1 = size_arr1.astype(int)
        size_arr2 = size_arr2.astype(int)

        size_arr = np.concatenate((size_arr1, size_arr2))

    state2_idx = np.where(states == state2)[0]            
    X_state2 = X[state2_idx]
    y_state2 = y[state2_idx]
    g_state2 = groups[state2_idx]

    # results = part4_worker(X, y, groups, states, years, 'LR', clf_dict, state, 0, 1000, X_state2, y_state2, g_state2)
    
    with Pool(processes=15) as pool:
        args = [(X, y, groups, states, years, clf, clf_dict, state, run, size, 
                 X_state2, y_state2, g_state2) for run in range(num_trials) for clf in clf_dict for size in size_arr]
        results = pool.starmap(part4_worker, args)
                
    import pdb; pdb.set_trace()
    new_data_pt = int(len(state_idx) * 0.8)
    results_df = pd.DataFrame(flatten(results))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    sns.lineplot(data=results_df, x='size', y='test_acc', hue='clf', ax=axes[0])
    axes[0].set_title("Accuracy with increasing training datasize",fontsize=14)
    axes[0].set_xscale('log')
    axes[0].axvline(x=new_data_pt, linestyle=':')
    sns.lineplot(data=results_df, x='size', y='EO', hue='clf', ax=axes[1])
    axes[1].set_title("EO with increasing training datasize",fontsize=14)
    axes[1].set_xscale('log')
    axes[1].axvline(x=new_data_pt, linestyle=':')
    
    sns.lineplot(data=results_df, x='size', y='worst', hue='clf', ax=axes[2])
    axes[2].set_title("Worst group perf with increasing training datasize",fontsize=14)
    axes[2].set_xscale('log')
    axes[2].axvline(x=new_data_pt, linestyle=':')
    plt.savefig(fname,bbox_inches='tight')
    
    csv_fname = fname.replace('.pdf','.csv').replace('figures/','csv/')
    results_df.to_csv(csv_fname)

def run_dip_experiments(data_name):
    """
    1: does performance degrade by year?
    2: does performance differ by state?
    3: does performance improve with more data for state 2?
    4: does the dip occur?
    """
    if data_name == 'mimic':
        LABEL1 = 'EW EMER.'
        LABEL2 = 'URGENT'
        LABEL_FIELD = 'admission_type'
        START_YEAR = 2008
        X, y, groups, states, years = get_mimic_data(LABEL1, LABEL2, LABEL_FIELD)
    elif data_name == 'yelp':
        LABEL1 = 'PA'
        LABEL2 = 'NJ' # NJ, AB, not IL bc no subgroups
        START_YEAR = 2006
        X, y, groups, states, years = get_yelp_data()
        # import pdb; pdb.set_trace()
    
    elif data_name == 'folktables':
        LABEL1 = 'CA'
        LABEL2 = 'SD'
        START_YEAR = 2014
        X, y, groups, states, years = get_folktables_data()
        
    clf_dict = {'LR':LogisticRegression, 
           # 'GB':GradientBoostingClassifier,
           #  'XGB': XGBClassifier
           # 'SVM':SVC,
           # 'NN':MLPClassifier
           }
    
    # part1(X, y, groups, states, years, clf_dict, start_year=START_YEAR, num_trials=5, fname='figures/%s_p1_years.pdf' % data_name)
    # part2(X, y, groups, states, years, clf_dict, LABEL1, num_trials=5, fname='figures/%s_p2_states.pdf' % data_name)
    part3(X, y, groups, states, years, clf_dict, LABEL1, LABEL2, num_trials=5, fname='figures/%s_p3_moredata.pdf' % data_name)
    part4(X, y, groups, states, years, clf_dict, LABEL1, LABEL2, num_trials=5, fname='figures/%s_p4_dip.pdf' % data_name)
    
    print('done! :D')
    return
    
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='mimic')
    args = parser.parse_args()
    
    run_dip_experiments(args.data)