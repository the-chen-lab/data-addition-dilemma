import numpy as np

def group_accuracy(correct_arr, group_arr, min_size=10): 
    g_acc_arr = [] 
    vals, counts = np.unique(group_arr, return_counts=True)
    for g, g_count in zip(vals, counts): 
        if g_count > min_size: 
            g_acc = np.mean(correct_arr[group_arr == g])
            g_acc_arr.append(g_acc)
    return g_acc_arr 