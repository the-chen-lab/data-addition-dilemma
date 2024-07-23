# run data scaling 
# python run_scaling.py --mixture --n_runs 1 --test_ratio 0.2 --ref_state 'SD' --state 'CA'
from folktables import ACSDataSource, ACSIncome
import metrics as mt
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.gaussian_process.kernels import RBF
from sklearn.pipeline import Pipeline
from sklearn import metrics

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
import argparse

# local libraries
import os
import sys
OTHER_STATES = ["SD", "NE", "IA", "MN", "OH", "PA", "MI", "TX", "LA", "GA", "FL", "CA", "SC", "WA", "MA"]
clf_list = ["LR", "GB", "XGB", "KNN", "NN"]
sys.path.append("..")


def add_data_filter(source_distribution, target_distribution, clf='RF', threshold=0.3):
    """Add a filter to the target distribution to make it more similar to the source distribution.
    """

    p = np.random.permutation(len(target_distribution))
    X_train = target_distribution[p][:len(source_distribution)]
    
    X = np.concatenate((source_distribution, X_train))
    Y = np.concatenate((np.zeros(len(source_distribution),), 
                        np.ones(len(X_train)),))
    # use random forest to predict whether a sample is from the source or target distribution 
    model = mt.model_choice(clf, X, Y)
    model.fit(X, Y)

    y_hat = model.predict_proba(target_distribution)[:, 1] # probability of class 0 (source distribution)
    keep = np.where(y_hat < threshold)[0]
    return keep

def get_data_for_state(data_dict,state, size, year, shuffle=True):
    """Get data for a state from the data dictionary."""
    X = data_dict[state][year]["x"]
    y = data_dict[state][year]["y"]
    g = data_dict[state][year]["g"]
    if shuffle: 
        p = np.random.permutation(len(X))
        X = X[p]
        y = y[p]
        g = g[p]
    return X[:size], y[:size], g[:size]

def run_data_scaling(mixture = False, 
                    n_runs = 1, 
                    test_ratio = 0.3, 
                    ref_state = ["CA"],
                    state = "SD", 
                    year = "2014",
                    seed = 0, 
                    filter_data=False, 
                    filter_threshold=0.5):

    data_dict = {} 
    print(ref_state)
    for s in ref_state +  [state]:
        data_dict[s] = {}
        data_source = ACSDataSource(survey_year=year, horizon="1-Year", survey="person")
        acs_data = data_source.get_data(states=[s], download=True)
        data_dict[s][year] = {}
        features, label, group = ACSIncome.df_to_numpy(acs_data)
        data_dict[s][year]["x"] = features
        data_dict[s][year]["y"] = label
        data_dict[s][year]["g"] = np.vectorize(mt.race_grouping.get)(group)

    # initialize generalized test set np arrays
    sample = 1000  
    x_gen_test = np.zeros((len(OTHER_STATES)*sample, data_dict[state][year]["x"].shape[1]))
    y_gen_test = np.zeros((len(OTHER_STATES)*sample,))
    g_gen_test = np.zeros((len(OTHER_STATES)*sample,))
 
    
    for i, other_state in enumerate(OTHER_STATES):

        data_source = ACSDataSource(survey_year=year, horizon="1-Year", survey="person")
        acs_data = data_source.get_data(states=[other_state], download=True)
        features, label, group = ACSIncome.df_to_numpy(acs_data)
        group = np.vectorize(mt.race_grouping.get)(group)
        incl = np.asarray(random.sample(range(len(features)), sample))

        x_gen_test[i*sample:(i+1)*sample] =  features[incl]
        y_gen_test[i*sample:(i+1)*sample] =  label[incl]
        g_gen_test[i*sample:(i+1)*sample] =  group[incl]

    results = []
    size_arr = [50, 100, 500, 1000, 2000, 4000, 8000, 12000, 14000, 16000]

    for run in range(n_runs):
        X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
            data_dict[state][year]["x"],
            data_dict[state][year]["y"],
            data_dict[state][year]["g"],
            test_size=test_ratio,
            random_state=seed+run,
        )
        # add reference state data
        if len(ref_state) > 1: 
            X_next, y_next, g_next = get_data_for_state(data_dict, ref_state[0], size_arr[-1], year, shuffle=True)
            for r_state in ref_state[1:]: 
                if r_state != state: 
                    X_state, y_state, g_state = get_data_for_state(data_dict, r_state, size_arr[-1], year, shuffle=True)
                    X_next = np.concatenate((X_next, X_state))
                    y_next = np.concatenate((y_next, y_state))
                    g_next = np.concatenate((g_next, g_state))
            p = np.random.permutation(len(X_next))
            X_next = X_next[p]
            y_next = y_next[p]
            g_next = g_next[p]  
        else: 
            X_next, y_next, g_next = get_data_for_state(data_dict, ref_state[0], size_arr[-1], year, shuffle=True)
        
        if filter_data:
            print("prior to filtering: ", len(X_next))
            selected_points = add_data_filter(
                                    np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1),
                                    np.concatenate((X_next, y_next.reshape(-1, 1)), axis=1), 
                                    "RF", 
                                    threshold=filter_threshold)
            X_next = X_next[selected_points]
            y_next = y_next[selected_points]
            g_next = g_next[selected_points]
            print("after filtering: ", len(X_next))


        X_joint = np.concatenate((X_train, X_next))
        y_joint = np.concatenate((y_train, y_next))
        g_joint = np.concatenate((group_train, g_next))
        if mixture:
            p = np.random.permutation(len(X_joint[: size_arr[-1]]))
            X_joint = X_joint[p]
            y_joint = y_joint[p]
            g_joint = g_joint[p]

        for clf in clf_list:
            for size in size_arr:
                if size > len(X_joint):
                    break
                X_train, X_eval, y_train, y_eval = train_test_split(
                    X_joint[:size],
                    y_joint[:size],
                    test_size=0.2
                )
                model = mt.model_choice(clf, X_joint[:size], y_joint[:size])

                model.fit(X_train, y_train)

                y_hat = model.predict(X_test)
                corr = y_hat == y_test
                acc_dict = mt.group_accuracy(corr, group_test)
                auc_dict = mt.group_auc(
                    y_test, model.predict_proba(X_test)[:, 1], group_test
                )
                fpr, tpr, thresholds = metrics.roc_curve(y_true=y_eval,
                                                         y_score=model.predict_proba(X_eval)[:, 1])
                opt_thresh = thresholds[np.argmax(tpr - fpr)]

                acc_ot_dict = mt.group_accuracy_ot(y_test, 
                                                   model.predict_proba(X_test)[:, 1], 
                                                   opt_thresh,
                                                   group_test)
                results.append(
                    {
                        "test_Accuracy": metrics.accuracy_score(y_hat, y_test),
                        "disp_Accuracy": max(acc_dict.values()) - min(acc_dict.values()),
                        "worst_g_Accuracy": min(acc_dict.values()),
                        "best_g_Accuracy": max(acc_dict.values()),
                        "nonwhite_Accuracy": acc_dict["non-white"],
                        "white_Accuracy": acc_dict["white"],
                        "black_Accuracy": acc_dict["black"] if "black" in acc_dict.keys() else np.nan,
                        # accuracy on same training ratio (eval set is not used for training)
                        "eval_Accuracy": metrics.accuracy_score(model.predict(X_eval), y_eval),
                        # accuracy on generalized test set of 10 different states 
                        "gen_Accuracy": metrics.accuracy_score(model.predict(x_gen_test), y_gen_test),
                        # test accuracy opt thresh
                        "test_Accuracy_OT": metrics.accuracy_score(model.predict_proba(X_test)[:, 1] > opt_thresh, y_test),
                        "disp_Accuracy_OT": max(acc_ot_dict.values()) - min(acc_ot_dict.values()),
                        "worst_g_Accuracy_OT": min(acc_ot_dict.values()),
                        "best_g_Accuracy_OT": max(acc_ot_dict.values()),
                        "nonwhite_Accuracy_OT": acc_ot_dict["non-white"],
                        "white_Accuracy_OT": acc_ot_dict["white"],
                        "black_Accuracy_OT": acc_ot_dict["black"] if "black" in acc_ot_dict.keys() else np.nan,
                        # AUC 
                        "test_AUC": metrics.roc_auc_score(
                            y_test, model.predict_proba(X_test)[:, 1]
                        ),
                        "disp_AUC": max(auc_dict.values()) - min(auc_dict.values()), 
                        "worst_g_AUC": min(auc_dict.values()),
                        "best_g_AUC": max(auc_dict.values()),
                        "nonwhite_AUC": auc_dict["non-white"],
                        "white_AUC": auc_dict["white"],
                        "black_AUC": auc_dict["black"] if "black" in auc_dict.keys() else np.nan,
                        "size": size,
                        "opt_thresh": opt_thresh,
                        "run": run,
                        "clf": clf,
                    }
                )

        results_df = pd.DataFrame(results)
        if filter_data: 
            filter_str = f"f{filter_threshold}"
        else: 
            filter_str = ""
        ref_states_str = "".join(ref_state)
        # create directory if one does not exist
        os.makedirs("../results", exist_ok=True)
        if mixture:
            results_df.to_csv(f"../results/scaling_mixture_a{state}_b{ref_states_str}_n{n_runs}_test{test_ratio}_s{seed}{filter_str}.csv")
        else:
            results_df.to_csv(f"../results/scaling_sequential_a{state}_b{ref_states_str}_n{n_runs}_test{test_ratio}_s{seed}{filter_str}.csv")
        

def scale_years(state="SD", 
                ref_year="2014", 
                n_runs=1, seed=0): 
    data_dict = {}
    data_dict[state] = {}
    year_arr = ["2014", "2015", "2016", "2017", "2018"] # years available for Folktables
    for year in year_arr:
        data_source = ACSDataSource(survey_year=year, horizon="1-Year", survey="person")
        acs_data = data_source.get_data(states=[state], download=True)
        data_dict[state][year] = {}
        features, label, group = ACSIncome.df_to_numpy(acs_data)
        group = np.vectorize(mt.race_grouping.get)(group)
        data_dict[state][year]["x"] = features
        data_dict[state][year]["y"] = label
        data_dict[state][year]["g"] = group

    size_arr = [50, 100, 500, 1000, 2000, 4000, 8000, 12000, 14000, 16000, 20000]
    mixture = True 
    results = [] 

    for run in range(n_runs): 
        X_joint, X_test, y_joint, y_test, g_joint, g_test = train_test_split(
            data_dict[state][ref_year]["x"], 
            data_dict[state][ref_year]["y"], 
            data_dict[state][ref_year]["g"], test_size=0.3, random_state=run
        )
        x_orig = len(X_joint)
        for year in year_arr: 
            if year != ref_year: 
                X_joint = np.concatenate((X_joint, data_dict[state][year]["x"]))
                y_joint = np.concatenate((y_joint, data_dict[state][year]["y"]))
                g_joint = np.concatenate((g_joint, data_dict[state][year]["g"]))

        if mixture: 
            p = np.random.permutation(len(X_joint))
            X_joint = X_joint[p]
            y_joint = y_joint[p]
            g_joint = g_joint[p]

        for size in size_arr: 
            for clf in clf_list:
                
                X_train, X_eval, y_train, y_eval = train_test_split(
                            X_joint[:size],
                            y_joint[:size],
                            test_size=0.2
                        )
                model = mt.model_choice(clf, X_joint[:size], y_joint[:size])

                model.fit(X_train, y_train)

                y_hat = model.predict(X_test)
                corr = y_hat == y_test
                acc_dict = mt.group_accuracy(corr, g_test)

                auc_dict = mt.group_auc(
                    y_test, model.predict_proba(X_test)[:, 1], g_test
                )

                fpr, tpr, thresholds = metrics.roc_curve(y_true=y_eval, 
                                                        y_score=model.predict_proba(X_eval)[:, 1])
                opt_thresh = thresholds[np.argmax(tpr - fpr)]

                acc_ot_dict = mt.group_accuracy_ot(y_test, 
                                                    model.predict_proba(X_test)[:, 1], 
                                                    opt_thresh,
                                                    g_test)
                
                results.append(
                    {
                        "test_Accuracy": metrics.accuracy_score(y_hat, y_test),
                        "disp_Accuracy": max(acc_dict.values()) - min(acc_dict.values()),
                        "worst_g_Accuracy": min(acc_dict.values()),
                        "best_g_Accuracy": max(acc_dict.values()),
                        "nonwhite_Accuracy": acc_dict["non-white"],
                        "white_Accuracy": acc_dict["white"],
                        "black_Accuracy": acc_dict["black"] if "black" in acc_dict.keys() else np.nan,
                        # Accuracy OT
                                        # test accuracy opt thresh
                        "test_Accuracy_OT": metrics.accuracy_score(model.predict_proba(X_test)[:, 1] > opt_thresh, y_test),
                        "disp_Accuracy_OT": max(acc_ot_dict.values()) - min(acc_ot_dict.values()),
                        "worst_g_Accuracy_OT": min(acc_ot_dict.values()),
                        "best_g_Accuracy_OT": max(acc_ot_dict.values()),
                        "nonwhite_Accuracy_OT": acc_ot_dict["non-white"],
                        "white_Accuracy_OT": acc_ot_dict["white"],
                        "black_Accuracy_OT": acc_ot_dict["black"] if "black" in acc_ot_dict.keys() else np.nan,
                
                        # AUC 
                        "test_AUC": metrics.roc_auc_score(
                            y_test, model.predict_proba(X_test)[:, 1]
                        ),
                        "disp_AUC": max(auc_dict.values()) - min(auc_dict.values()), 
                        "worst_g_AUC": min(auc_dict.values()),
                        "best_g_AUC": max(auc_dict.values()),
                        "nonwhite_AUC": auc_dict["non-white"],
                        "white_AUC": auc_dict["white"],
                        "black_AUC": auc_dict["black"] if "black" in auc_dict.keys() else np.nan,
                        "size": size,
                        "run": run,
                        "clf": clf,
                    }
                )

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"../results/{state}_scaling_n{n_runs}.csv")
    return 


def scale_simple(state="CA", n_runs=1): 
    data_source = ACSDataSource(survey_year="2014", horizon="1-Year", survey="person")
    acs_data = data_source.get_data(states=[state], download=True)
    features, label, group = ACSIncome.df_to_numpy(acs_data)
    group = np.vectorize(mt.race_grouping.get)(group)

    size_arr = [100, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 160000]
    results = [] 
    for run in range(n_runs): 
        X_joint, X_test, y_joint, y_test, g_train, g_test = train_test_split(
            features, label, group, test_size=0.1, random_state=run
        )
        for size in size_arr: 
            print(size)
            for clf in clf_list:
                
                X_train, X_eval, y_train, y_eval = train_test_split(
                            X_joint[:size],
                            y_joint[:size],
                            test_size=0.2
                        )
                model = mt.model_choice(clf, X_joint[:size], y_joint[:size])

                model.fit(X_train, y_train)

                y_hat = model.predict(X_test)
                corr = y_hat == y_test
                acc_dict = mt.group_accuracy(corr, g_test)

                auc_dict = mt.group_auc(
                    y_test, model.predict_proba(X_test)[:, 1], g_test
                )

                fpr, tpr, thresholds = metrics.roc_curve(y_true=y_eval, 
                                                        y_score=model.predict_proba(X_eval)[:, 1])
                opt_thresh = thresholds[np.argmax(tpr - fpr)]

                acc_ot_dict = mt.group_accuracy_ot(y_test, 
                                                    model.predict_proba(X_test)[:, 1], 
                                                    opt_thresh,
                                                    g_test)
                
                results.append(
                    {
                        "test_Accuracy": metrics.accuracy_score(y_hat, y_test),
                        "disp_Accuracy": max(acc_dict.values()) - min(acc_dict.values()),
                        "worst_g_Accuracy": min(acc_dict.values()),
                        "best_g_Accuracy": max(acc_dict.values()),
                        "nonwhite_Accuracy": acc_dict["non-white"],
                        "white_Accuracy": acc_dict["white"],
                        "black_Accuracy": acc_dict["black"] if "black" in acc_dict.keys() else np.nan,
                        # Accuracy OT
                                        # test accuracy opt thresh
                        "test_Accuracy_OT": metrics.accuracy_score(model.predict_proba(X_test)[:, 1] > opt_thresh, y_test),
                        "disp_Accuracy_OT": max(acc_ot_dict.values()) - min(acc_ot_dict.values()),
                        "worst_g_Accuracy_OT": min(acc_ot_dict.values()),
                        "best_g_Accuracy_OT": max(acc_ot_dict.values()),
                        "nonwhite_Accuracy_OT": acc_ot_dict["non-white"],
                        "white_Accuracy_OT": acc_ot_dict["white"],
                        "black_Accuracy_OT": acc_ot_dict["black"] if "black" in acc_ot_dict.keys() else np.nan,
                
                        # AUC 
                        "test_AUC": metrics.roc_auc_score(
                            y_test, model.predict_proba(X_test)[:, 1]
                        ),
                        "disp_AUC": max(auc_dict.values()) - min(auc_dict.values()), 
                        "worst_g_AUC": min(auc_dict.values()),
                        "best_g_AUC": max(auc_dict.values()),
                        "nonwhite_AUC": auc_dict["non-white"],
                        "white_AUC": auc_dict["white"],
                        "black_AUC": auc_dict["black"] if "black" in auc_dict.keys() else np.nan,
                        "size": size,
                        "run": run,
                        "clf": clf,
                    }
                )

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"../results/{state}_scaling_n{n_runs}.csv")
    return 

def main():
    parser = argparse.ArgumentParser(description="Run Data Scaling")

    # Add arguments for the function
    parser.add_argument('--mixture', dest='mixture', action='store_true',
                        help='Flag to enable mixture.')
    parser.add_argument('--sequential', dest='sequential', action='store_true',
                    help='Flag to enable sequential.')
    parser.add_argument('--scale_years', dest='scale_years', action='store_true',
                    help='Flag to enable scale years.')
    parser.add_argument('--scale_simple', dest='scale_simple', action='store_true',
                    help='Flag to enable scale simple.')
    parser.set_defaults(mixture=False)
    parser.add_argument('--n_runs', type=int, default=1,
                        help='Number of runs.')
    parser.add_argument('--test_ratio', type=float, default=0.3,
                        help='Test ratio.')
    parser.add_argument('--ref_state', nargs='*', type=str, default=["CA"],
                        help='Reference state.')
    parser.add_argument('--state', type=str, default="SD",
                        help='State.')
    parser.add_argument('--year', type=str, default="2014",
                        help='Year.')
    parser.add_argument('--seed', type=int, default=0,
                        help='random_seed')
    parser.add_argument('--filter_data', dest='filter_data', action='store_true')
    parser.add_argument('--filter_threshold', type=float, default=0.5)

    # Parse the arguments
    args = parser.parse_args()

    if args.mixture or args.sequential: 
        # Call the function with parsed arguments
        run_data_scaling(mixture=args.mixture,
                        n_runs=args.n_runs,
                        test_ratio=args.test_ratio,
                        ref_state=args.ref_state,
                        state=args.state,
                        year=args.year,
                        seed=args.seed, 
                        filter_data=args.filter_data,
                        filter_threshold=args.filter_threshold)
    if args.scale_years: 
        scale_years(state=args.state,
                    ref_year=args.year,
                    n_runs=args.n_runs)
    if args.scale_simple: 
        scale_simple(state=args.state,
                    n_runs=args.n_runs)

if __name__ == "__main__":
    main()
