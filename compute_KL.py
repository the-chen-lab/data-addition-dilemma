# KL Distribution Check
# python your_script_name.py --mixture --n_runs 1 --n_samples 5000 --year 2014 --test_ratio 0.3

#from folktables import ACSDataSource, ACSEmployment, ACSIncome
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
import os 
import argparse 

import sys
sys.path.append("..")
import metrics as mt
import pdb

# CONSTS
hospital_ids = [73, 264, 420, 243, 338, 443, 199, 458, 300, 188, 252, 167]
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


def get_hospital(hid, split='train', max_samples=None, sample_ratio=1, rand_seed=42): 
    log_dir = f'/home/ubuntu/projects/more-data-more-problems/yaib_logs/eicu/Mortality24/LogisticRegression/'
    file_name =f'train{hid}-test{hid}/data.npz'
    hos = np.load(os.path.join(log_dir, file_name), allow_pickle=True)
    x = hos[split].item()['features']
    y = hos[split].item()['labels']
    xy = np.concatenate((x, y.reshape(-1, 1)), axis=1)
    if sample_ratio < 1: 
        rng = np.random.default_rng(rand_seed)
        ind = rng.choice(len(x), size=int(len(x)*sample_ratio), replace=False)
        return x[ind], y[ind], xy[ind]
    elif max_samples is not None: 
        if len(x) > max_samples: 
            rng = np.random.default_rng(rand_seed)
            ind = rng.choice(len(x), size=int(max_samples), replace=False)
            return x[ind], y[ind], xy[ind]
        else: 
            return x, y, xy
    else: 
        return x, y, xy

def fit_general_density(hids, split='train', max_samples=5000, n_components=3):
    # fit stratified sample density
    num_hospitals = len(hids)
    samples_per_hos = int(max_samples / num_hospitals)

    x_all = []
    xy_all = []

    for h in hids:
        x, y, xy = get_hospital(h, split=split)

        # Sample from the hospital data
        random_indices = np.random.choice(len(x), size=samples_per_hos, replace=False)
        x_sampled = x[random_indices]
        xy_sampled = xy[random_indices]

        # Append the sampled data to the overall arrays
        x_all.append(x_sampled)
        xy_all.append(xy_sampled)

    # Concatenate the sampled data from all hospitals
    x_all = np.concatenate(x_all, axis=0)
    xy_all = np.concatenate(xy_all, axis=0)
    print(f"fitting overall density function with {len(x_all)} samples from {len(hids)}")
    cx, _ = mt.init_density_scale(x_all, n_components=n_components)
    cxy, _ = mt.init_density_scale(xy_all, n_components=n_components)
    return cx, cxy
    
def run_hospital_kl(n_runs=5, n_samples=2000, n_components=3): 
    KL_x = np.zeros((n_runs, len(hospital_ids), len(hospital_ids)))
    KL_xy = np.zeros((n_runs, len(hospital_ids), len(hospital_ids)))
    results = {} 
    for run in range(n_runs):
        # cx, cxy = fit_general_density(hospital_ids, 
        #                       max_samples=10000,
        #                       n_components=n_components)
        print(f"iter {run}")
        for i, h1 in enumerate(hospital_ids): 
            x, y, xy = get_hospital(h1, sample_ratio=0.9, rand_seed=run)
            # test set 
            # pdb.set_trace()
         
            for j, h2 in enumerate(hospital_ids): 
                if i != j: 
                    print(f"computing {h1} {h2}") 
                    x2, y2, xy2 = get_hospital(h2, sample_ratio=0.9, rand_seed=run)
                    cx, _ = mt.init_density_scale(np.concatenate((x, x2), axis=0), n_components=n_components)
                    cxy, _ = mt.init_density_scale(np.concatenate((xy, xy2), axis=0), n_components=n_components)
                    x2, y2, xy2 = x2[:n_samples], y2[:n_samples], xy2[:n_samples]
                    # train set
                    # already shuffled
                    x, y, xy = x[:n_samples], y[:n_samples], xy[:n_samples]
                    pkdex = mt.init_density(x, cx) 
                    pkdexy = mt.init_density(xy, cxy)    
            
                    # test set
                    qkdex = mt.init_density(x2, cx)
                    qkdexy = mt.init_density(xy2, cxy)
                    #pdb.set_trace()
                    KL_x[run, i, j] = mt.entropy_input(x, pkdex, qkdex, cx)
                    KL_xy[run, i, j] = mt.entropy_input(xy, pkdexy, qkdexy, cxy)
                    print(f"computing {h1} {h2}, kl_xy{KL_xy[run, i, j]}") 
        results['KL_x'] = KL_x
        results['KL_xy'] = KL_xy
        np.savez(f"YAIB/results/distances/KL-n{n_samples}-c{n_components}-pair.npz", **results)
    return 

def main():
    parser = argparse.ArgumentParser(description="Run KL Check")
    
    # Add arguments for the function
    parser.add_argument('--dataset', type=str, default='eicu',
                        help='dataset')
    parser.add_argument('--mixture', dest='mixture', action='store_true',
                        help='Flag to enable mixture.')
    parser.set_defaults(mixture=False)  # default value for mixture
    parser.add_argument('--n_runs', type=int, default=1,
                        help='Number of runs.')
    parser.add_argument('--n_samples', type=int, default=3000,
                        help='Number of samples.')
    parser.add_argument('--n_components', type=int, default=3,
                        help='Number of PCA Components')
    parser.add_argument('--year', type=str, default="2014",
                        help='Year.')
    parser.add_argument('--test_ratio', type=float, default=0.3,
                        help='Test ratio.')

    # Parse the arguments
    args = parser.parse_args()
    
    # Call the function using parsed arguments
    if args.dataset == 'folktables': 
        run_kl_check(mixture=args.mixture,
                     n_runs=args.n_runs,
                     n_samples=args.n_samples,
                     year=args.year,
                     test_ratio=args.test_ratio)
    elif args.dataset == 'eicu': 
        run_hospital_kl(n_runs = args.n_runs, 
                       n_samples=args.n_samples, 
                       n_components=args.n_components)

if __name__ == "__main__":
    main()
