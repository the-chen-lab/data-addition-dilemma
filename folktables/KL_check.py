# KL Distribution Check
# python KL_check.py --mixture --n_runs 1 --n_samples 5000 --year 2014 --test_ratio 0.3

from folktables import ACSDataSource, ACSEmployment, ACSIncome, metrics as mt
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
sys.path.append("..")


def run_kl_check(mixture=False, 
                 n_runs=1, 
                 n_samples=3000,
                 year="2014",
                 test_ratio = 0.3): 

    data_dict = {} 
    for state in ["CA", "SD"]:
        data_dict[state] = {}
        data_source = ACSDataSource(survey_year=year, horizon="1-Year", survey="person")
        acs_data = data_source.get_data(states=[state], download=True)
        data_dict[state][year] = {}
        features, label, group = ACSIncome.df_to_numpy(acs_data)
        data_dict[state][year]["x"] = features
        data_dict[state][year]["y"] = label
        data_dict[state][year]["g"] = group
    
    results = []
    state = "SD"
    size_arr = [500, 1000, 2000, 3000, 4000, 8000, 12000, 14000, 16000]
    
    for run in range(n_runs):
        X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
            data_dict[state][year]["x"],
            data_dict[state][year]["y"],
            data_dict[state][year]["g"],
            test_size=test_ratio,
            random_state=run,
        )

        X_joint = np.concatenate((X_train, data_dict["CA"][year]["x"]))
        y_joint = np.concatenate((y_train, data_dict["CA"][year]["y"]))
        g_joint = np.concatenate((group_train, data_dict["CA"][year]["g"]))

        # joint density kernel
        joint_xy_ref = np.concatenate((X_joint, y_joint.reshape(-1, 1)), axis=1)

        incl = np.asarray(random.sample(range(len(joint_xy_ref)), n_samples))

        cx, _ = mt.init_density_scale(X_joint[incl])

        cxy, _ = mt.init_density_scale(joint_xy_ref[incl])

        # reference density using test set
        qkdex = mt.init_density(X_test, cx)
        qkdexy = mt.init_density(
            np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1), cxy
        )

        if mixture:
            p = np.random.permutation(len(X_joint[: size_arr[-1]]))
            X_joint = X_joint[p]
            y_joint = y_joint[p]
            g_joint = g_joint[p]

        for size in size_arr:
            joint_xy = np.concatenate(
                (X_joint[:size], y_joint[:size].reshape(-1, 1)), axis=1
            )

            if size > n_samples:
                incl = np.asarray(random.sample(range(size), n_samples))
            else:
                incl = range(size)

            pkdexy = mt.init_density(joint_xy[:size][incl], cxy)
            pkdex = mt.init_density(X_joint[:size][incl], cx)

            results.append(
                {
                    "KL_x": mt.entropy_input(X_joint[:size][incl], pkdex, qkdex, cx),
                    "KL_xy": mt.entropy_input(joint_xy[:size][incl], pkdexy, qkdexy, cxy),
                    "size": size,
                    "run": run,
                }
            )

    results_df = pd.DataFrame(results)
    if mixture:
        results_df.to_csv(f"../results/KL_mixture_{state}_n{n_runs}_samples{n_samples}.csv")
    else:
        results_df.to_csv(f"../results/KL_sequential_{state}_n{n_runs}_samples{n_samples}.csv")
    
    return 


def main():
    parser = argparse.ArgumentParser(description="Run KL Check")
    
    # Add arguments for the function
    parser.add_argument('--mixture', dest='mixture', action='store_true',
                        help='Flag to enable mixture.')
    parser.set_defaults(mixture=False)  # default value for mixture
    parser.add_argument('--n_runs', type=int, default=1,
                        help='Number of runs.')
    parser.add_argument('--n_samples', type=int, default=3000,
                        help='Number of samples.')
    parser.add_argument('--year', type=str, default="2014",
                        help='Year.')
    parser.add_argument('--test_ratio', type=float, default=0.3,
                        help='Test ratio.')

    # Parse the arguments
    args = parser.parse_args()
    
    # Call the function using parsed arguments
    run_kl_check(mixture=args.mixture,
                 n_runs=args.n_runs,
                 n_samples=args.n_samples,
                 year=args.year,
                 test_ratio=args.test_ratio)


if __name__ == "__main__":
    main()
