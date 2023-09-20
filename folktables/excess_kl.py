#Excess KL
# python excess_kl.py --n_samples 5000 --n_runs 3

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
import scipy.stats

import argparse 

import sys
import os 
sys.path.append("..")
import metrics as mt

STATES = ["CA", "OH", "DE", "AK", "HI", "SD", "ND", "PA", "MI", "GA", "MS"]
def run_states(n_runs=1): 

    year = "2014"
    data_dict = {}
    for state in STATES:
        data_dict[state] = {}
        data_source = ACSDataSource(survey_year=year, horizon="1-Year", survey="person")
        acs_data = data_source.get_data(states=[state], download=True)
        data_dict[state][year] = {}
        features, label, group = ACSIncome.df_to_numpy(acs_data)
        data_dict[state][year]["x"] = features
        data_dict[state][year]["y"] = label
        data_dict[state][year]["g"] = np.vectorize(mt.race_grouping.get)(group)
    
    results = []
    for clf in mt.clf_dict:
        for i in range(n_runs):
            ref_state = "CA"
            X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
                data_dict[ref_state][year]["x"],
                data_dict[ref_state][year]["y"],
                data_dict[ref_state][year]["g"],
                test_size=0.2,
                random_state=i,
            )

            model = mt.model_choice(clf, X_train, y_train)

            model.fit(X_train, y_train)

            y_hat = model.predict(X_test)
            corr = y_hat == y_test
            g_acc_dict = mt.group_accuracy(corr, group_test)
            g_auc_dict = mt.group_auc(y_test, model.predict_proba(X_test)[:, 1], group_test)

            results.append(
                {
                    "year": year,
                    "test_Accuracy": metrics.accuracy_score(y_hat, y_test),
                    "disp_Accuracy": max(g_acc_dict.values()) - min(g_acc_dict.values()),
                    "worst_g_Accuracy": min(g_acc_dict.values()),
                    "test_AUC": metrics.roc_auc_score(
                        y_test, model.predict_proba(X_test)[:, 1]
                    ),
                    "disp_AUC": max(g_auc_dict.values()) - min(g_auc_dict.values()),
                    "worst_g_AUC": min(g_auc_dict.values()),
                    "size": len(y_train),
                    "run": i,
                    "state": ref_state,
                    "clf": clf,
                }
            )

            for state in STATES:
                if state != ref_state:
                    (
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        group_train,
                        group_test,
                    ) = train_test_split(
                        data_dict[state][year]["x"],
                        data_dict[state][year]["y"],
                        data_dict[state][year]["g"],
                        test_size=0.4,
                        random_state=i,
                    )
                    y_hat = model.predict(X_test)
                    corr = y_hat == y_test
                    g_acc_dict = mt.group_accuracy(corr, group_test)
                    g_auc_dict = mt.group_auc(
                        y_test, model.predict_proba(X_test)[:, 1], group_test
                    )

                    results.append(
                        {
                            "year": year,
                            "test_Accuracy": metrics.accuracy_score(y_hat, y_test),
                            "disp_Accuracy": max(g_acc_dict.values())
                            - min(g_acc_dict.values()),
                            "worst_g_Accuracy": min(g_acc_dict.values()),
                            "test_AUC": metrics.roc_auc_score(
                                y_test, model.predict_proba(X_test)[:, 1]
                            ),
                            "disp_AUC": max(g_auc_dict.values()) - min(g_auc_dict.values()),
                            "worst_g_AUC": min(g_auc_dict.values()),
                            "size": len(y_train),
                            "run": i,
                            "state": state,
                            "clf": clf,
                        }
                    )
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"../results/states_results.csv")
    return results_df


def kl_accuracy(states_df, n_runs=1, n_samples=3000):
    states_mean = states_df.groupby(["state", "clf"]).mean()
    ref_state = "CA"
    samples = 3000
    year = "2014"
    data_dict = {}
    for state in STATES:
        data_dict[state] = {}
        data_source = ACSDataSource(survey_year=year, horizon="1-Year", survey="person")
        acs_data = data_source.get_data(states=[state], download=True)
        data_dict[state][year] = {}
        features, label, group = ACSIncome.df_to_numpy(acs_data)
        data_dict[state][year]["x"] = features
        data_dict[state][year]["y"] = label
        data_dict[state][year]["g"] = np.vectorize(mt.race_grouping.get)(group)

    for i in range(n_runs):
        kl_df = pd.DataFrame()
        for state in STATES:
            print(state)
            if state != ref_state:
                joint_xy_ref = np.concatenate(
                    (
                        data_dict[ref_state][year]["x"],
                        data_dict[ref_state][year]["y"].reshape(-1, 1),
                    ),
                    axis=1,
                )

                joint_xy_state = np.concatenate(
                    (
                        data_dict[state][year]["x"],
                        data_dict[state][year]["y"].reshape(-1, 1),
                    ),
                    axis=1,
                )
                joint_xy = np.concatenate((joint_xy_ref, joint_xy_state), axis=0)
                joint_x = np.concatenate(
                    (data_dict[ref_state][year]["x"], data_dict[state][year]["x"]), axis=0
                )

                incl = np.asarray(random.sample(range(len(joint_xy)), n_samples))
                cx, _ = mt.init_density_scale(joint_x[incl])
                cxy, _ = mt.init_density_scale(joint_xy[incl])

                incl = np.asarray(random.sample(range(len(joint_xy_ref)), n_samples))
                pkdex = mt.init_density(data_dict[ref_state][year]["x"][incl], cx)
                pkdexy = mt.init_density(joint_xy_ref[incl], cxy)

                diff_df = states_mean.loc[ref_state] - states_mean.loc[state]

                # sample
                incl = np.asarray(random.sample(range(len(joint_xy_state)), n_samples))
                qkdexy = mt.init_density(joint_xy_state[incl], cxy)
                qkdex = mt.init_density(data_dict[state][year]["x"][incl], cx)

                diff_df["KL_x"] = mt.entropy_input(
                    data_dict[ref_state][year]["x"][incl], pkdex, qkdex, cx
                )
                diff_df["KL_xy"] = mt.entropy_input(joint_xy_ref[incl], pkdexy, qkdexy, cxy)
                diff_df["state"] = state
                diff_df["iter"] = i
                kl_df = pd.concat((kl_df, diff_df))
    
    kl_df.to_csv(f"../results/kl_shift_n{n_runs}_s{n_samples}.csv")
    return 
                

def main():
    parser = argparse.ArgumentParser(description="Run KL Check")
    
    # Add arguments for the function
    parser.add_argument('--state_kl', dest='state_kl', action='store_true',
                        help='Flag to enable computing state kl.')
    parser.set_defaults(mixture=False)  # default value for mixture
    parser.add_argument('--n_runs', type=int, default=1,
                        help='Number of runs.')
    parser.add_argument('--n_samples', type=int, default=3000,
                        help='Number of samples.')
    
    # Parse the arguments
    args = parser.parse_args()
    
    if os.path.isfile("../results/states_results.csv"):
        states_df = pd.read_csv("../results/states_results.csv")
    else: 
        states_df = run_states(n_runs=args.n_runs)
    

    # Call the function using parsed arguments
    if args.state_kl:
        kl_accuracy(states_df, n_runs=args.n_runs, n_samples=args.n_samples)

if __name__ == "__main__":
    main()
