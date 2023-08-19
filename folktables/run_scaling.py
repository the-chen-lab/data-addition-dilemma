# run data scaling 

from folktables import ACSDataSource, ACSEmployment, ACSIncome
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

import scipy.stats

# local libraries
import sys

sys.path.append("..")
import metrics as mt

def run_data_scaling(mixture = False, 
                    n_runs = 1, 
                    test_ratio = 0.3, 
                    ref_state = "CA",
                    state = "SD", 
                    year = "2014",
                    seed = 0):

    data_dict = {} 
    for state in [ref_state, state]:
        data_dict[state] = {}
        data_source = ACSDataSource(survey_year=year, horizon="1-Year", survey="person")
        acs_data = data_source.get_data(states=[state], download=True)
        data_dict[state][year] = {}
        features, label, group = ACSIncome.df_to_numpy(acs_data)
        data_dict[state][year]["x"] = features
        data_dict[state][year]["y"] = label
        data_dict[state][year]["g"] = group
        
    
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

        X_joint = np.concatenate((X_train, data_dict[ref_state][year]["x"]))
        y_joint = np.concatenate((y_train, data_dict[ref_state][year]["y"]))
        g_joint = np.concatenate((group_train, data_dict[ref_state][year]["g"]))
        if mixture:
            p = np.random.permutation(len(X_joint[: size_arr[-1]]))
            X_joint = X_joint[p]
            y_joint = y_joint[p]
            g_joint = g_joint[p]

        for clf in mt.clf_dict.keys():
            for size in size_arr:
                model = mt.model_choice(clf, X_joint[:size], y_joint[:size])

                model.fit(X_joint[:size], y_joint[:size])

                y_hat = model.predict(X_test)
                corr = y_hat == y_test
                g_acc_arr, acc_dict = mt.group_accuracy(corr, group_test)
                g_auc_arr, auc_dict = mt.group_auc(
                    y_test, model.predict_proba(X_test)[:, 1], group_test
                )

                train_acc = model.score(X_train, y_train)

                test_acc = model.score(X_test, y_test)

                results.append(
                    {
                        "train_acc": train_acc,
                        "test_Accuracy": metrics.accuracy_score(y_hat, y_test),
                        "disp_Accuracy": max(g_acc_arr) - min(g_acc_arr),
                        "worst_g_Accuracy": min(g_acc_arr),
                        "best_g_Accuracy": max(g_acc_arr),
                        "test_AUC": metrics.roc_auc_score(
                            y_test, model.predict_proba(X_test)[:, 1]
                        ),
                        "disp_AUC": max(g_auc_arr) - min(g_auc_arr)
                        if len(g_auc_arr) > 0
                        else 0,
                        "worst_g_AUC": min(g_auc_arr) if len(g_auc_arr) > 0 else 0,
                        "best_g_AUC": max(g_auc_arr) if len(g_auc_arr) > 0 else 0,
                        "nonwhite_Accuracy": acc_dict["non-white"],
                        "white_Accuracy": acc_dict["white"],
                        "black_Accuracy": acc_dict["black"] if "black" in acc_dict.keys() else np.nan,
                        "nonwhite_AUC": auc_dict["non-white"],
                        "white_AUC": auc_dict["white"],
                        "black_AUC": auc_dict["black"] if "black" in auc_dict.keys() else np.nan,
                        "size": size,
                        "run": run,
                        "clf": clf,
                    }
                )

        results_df = pd.DataFrame(results)
        if mixture:
            results_df.to_csv(f"../results/scaling_mixture_{state}_n{n_runs}_test{test_ratio}_s{seed}.csv")
        else:
            results_df.to_csv(f"../results/scaling_sequential_{state}_n{n_runs}_test{test_ratio}_s{seed}.csv")
        
def main():
    parser = argparse.ArgumentParser(description="Run Data Scaling")

    # Add arguments for the function
    parser.add_argument('--mixture', dest='mixture', action='store_true',
                        help='Flag to enable mixture.')
    parser.set_defaults(mixture=False)
    parser.add_argument('--n_runs', type=int, default=1,
                        help='Number of runs.')
    parser.add_argument('--test_ratio', type=float, default=0.3,
                        help='Test ratio.')
    parser.add_argument('--ref_state', type=str, default="CA",
                        help='Reference state.')
    parser.add_argument('--state', type=str, default="SD",
                        help='State.')
    parser.add_argument('--year', type=str, default="2014",
                        help='Year.')
    parser.add_argument('--seed', type=int, default=0,
                        help='random_seed')

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with parsed arguments
    run_data_scaling(mixture=args.mixture,
                     n_runs=args.n_runs,
                     test_ratio=args.test_ratio,
                     ref_state=args.ref_state,
                     state=args.state,
                     year=args.year,
                     seed=args.seed)

if __name__ == "__main__":
    main()
