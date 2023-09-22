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

# more states 
# near SD : NE, IA, MN
# mid west: OH, PA, MI
# south: TX, LA, GA, FL
# coast: CA, WA, MA

STATES = ["SD", "NE", "IA", "MN", "OH", "PA", "MI", "TX", "LA", "GA", "FL", "CA", "SC", "WA", "MA"]
clf_list = ["LR", "GB", "XGB", "KNN", "NN"]
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

                # estimate kernel density from joint distribution of both both states
                # results are similar with just the reference state
                incl = np.asarray(random.sample(range(len(joint_xy)), n_samples))
                cx, _ = mt.init_density_scale(joint_x[incl])
                cxy, _ = mt.init_density_scale(joint_xy[incl])

                incl = np.asarray(random.sample(range(len(joint_xy_ref)), n_samples))
                pkdex = mt.init_density(data_dict[ref_state][year]["x"][incl], cx)
                pkdexy = mt.init_density(joint_xy_ref[incl], cxy)

                # change in accuracy (negative)
                diff_df = states_mean.loc[state] - states_mean.loc[ref_state] 

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
                

def excess_kl(n_runs=1, n_samples=3000, large_size=12000):
    results = []

    year = "2014"
    # init kde and transform with SD data
    for run in range(n_runs):
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

        # must estimate kde based on state (e.g. south dakota)
        state = "SD"
        X_train, X_test, y_train, y_test, group_train, _, = train_test_split(
                        data_dict[state][year]["x"],
                        data_dict[state][year]["y"],
                        data_dict[state][year]["g"],
                        test_size=0.25,
        )
        
        joint_xy = np.concatenate(
            (X_train, y_train.reshape(-1, 1)),
            axis=1,
        )
        incl = np.asarray(random.sample(range(len(joint_xy)), n_samples))

        size_arr = [len(X_train), large_size] # change these numbers based on state test set 
        cx, _ = mt.init_density_scale(X_train[incl])
        cxy, _ = mt.init_density_scale(joint_xy[incl]) 
        qkdex = mt.init_density(X_test, cx)
        qkdexy = mt.init_density(np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1), cxy)

        print(run)
        random.seed(run)
        for extra_state in STATES:
            if extra_state != state:
                p = np.random.permutation(len(data_dict[extra_state][year]["x"][: size_arr[-1]]))
                # permute data from each extra_state before combining with SD
                X_joint = np.concatenate((X_train, data_dict[extra_state][year]["x"][p]))
                y_joint = np.concatenate((y_train, data_dict[extra_state][year]["y"][p]))

                xy_joint = np.concatenate((X_joint, y_joint.reshape(-1, 1)), axis=1)

                for size in size_arr:
                # sub sample
                    if size < len(X_joint): 
                        incl = np.asarray(random.sample(range(size), n_samples))
                    else:
                        incl = np.asarray(random.sample(range(len(X_joint)), n_samples))

                    pkdex = mt.init_density(X_joint[:size][incl], cx)
                    pkdexy = mt.init_density(xy_joint[:size][incl], cxy)

                    kl_x = mt.entropy_input(X_joint[:size][incl], pkdex, qkdex, cx)
                    kl_xy = mt.entropy_input(xy_joint[:size][incl], pkdexy, qkdexy, cxy)

                    for clf in mt.clf_dict.keys():
                        model = mt.model_choice(clf, X_joint[:size], y_joint[:size])

                        model.fit(X_joint[:size], y_joint[:size])

                        train_acc = model.score(X_train, y_train)

                        test_acc = model.score(X_test, y_test)

                        results.append(
                            {
                                "train_acc": train_acc,
                                "test_acc": test_acc,
                                "kl_testx": kl_x,
                                "kl_testxy": kl_xy,
                                "size": size,
                                "run": run,
                                "clf": clf,
                                "extra_state": extra_state,
                            }
                        )

        results_df = pd.DataFrame(results)

        q1_results = results_df[results_df["size"] == len(X_train)].reset_index()[
            ["test_acc", "clf", "extra_state", "kl_testx", "kl_testxy"]
        ]
        q2_results = results_df[results_df["size"] == large_size].reset_index()[
            ["test_acc", "clf", "extra_state", "kl_testx", "kl_testxy"]
        ]
        diff_results = pd.DataFrame()
        # accuracy drop n_small - n_large
        diff_results["acc_diff"] = q2_results["test_acc"] - q1_results["test_acc"]
        # excess kl n_large - n_small
        diff_results["klx_diff"] = q2_results["kl_testx"] - q1_results["kl_testx"]
        diff_results["klxy_diff"] = q2_results["kl_testxy"] - q1_results["kl_testxy"]
        diff_results["clf"] = q1_results["clf"]
        diff_results["extra_state"] = q1_results["extra_state"]
        diff_results.to_csv(f"../results/excess_kl_n{n_runs}_s{n_samples}_l{large_size}.csv")
    return 


def main():
    parser = argparse.ArgumentParser(description="Run KL Check")
    
    # Add arguments for the function
    parser.add_argument('--state_kl', dest='state_kl', action='store_true',
                        help='Flag to enable computing state kl.')
    parser.add_argument('--excess_kl', dest='excess_kl', action='store_true',
                        help='Flag to enable computing excess kl.')
    parser.set_defaults(mixture=False)  # default value for mixture
    parser.add_argument('--n_runs', type=int, default=1,
                        help='Number of runs.')
    parser.add_argument('--large_size', type=int, default=12000)
    parser.add_argument('--n_samples', type=int, default=3000,
                        help='Number of samples.')
    
    # Parse the arguments
    args = parser.parse_args()
    

    

    # Call the function using parsed arguments
    if args.state_kl:
        if os.path.isfile("../results/states_results.csv"):
            states_df = pd.read_csv("../results/states_results.csv")
        else: 
            states_df = run_states(n_runs=args.n_runs)
    
        kl_accuracy(states_df, n_runs=args.n_runs, n_samples=args.n_samples)
    if args.excess_kl:
        excess_kl(n_runs=args.n_runs, n_samples=args.n_samples, large_size=args.large_size)

if __name__ == "__main__":
    main()
