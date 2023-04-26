import torch
from sklearn.metrics import roc_auc_score
import numpy as np

def getCItd(y_pred, o_arr, t_arr, t_step=1, flag_hot=False):
    # res = model.predict(ds, steps=steps, verbose=1)
    if len(y_pred) != len(t_arr):
        print('length: res %d, label %d' % (len(y_pred), len(t_arr)))
        # if len(res) > len(y_true):
        #     res = res[:len(y_true)]
    assert len(y_pred) == len(t_arr)
    o_arr = np.array(o_arr).astype(np.float)
    t_arr = np.array(t_arr).astype(np.float)
    
    o_arr = o_arr / t_step
    t_arr = t_arr / t_step

    o_arr = o_arr.astype(np.int)
    t_arr = t_arr.astype(np.int)
    

    t_max = np.max(t_arr)
    t_max = int(np.ceil(t_max))
    t_max = min(t_max, len(y_pred[0]))
    w_pred = np.cumsum(y_pred, axis=-1)
    flag_dead = t_arr < o_arr
    tot_pair = 0
    ret = 0

    if flag_hot:
        t_arr = np.array([min(o, t) for o, t in zip(o_arr, t_arr)])

    for t in range(t_max):
        flag_lt = (t_arr > t)
        flag_et = ((t_arr == t) & flag_dead)
        n_et = np.sum(flag_et)
        n_lt = np.sum(flag_lt)
        if n_et == 0 or n_lt == 0:
            continue
        tmp_w_lt = w_pred[flag_lt][:, t]
        tmp_w_et = w_pred[flag_et][:, t]
        tmp_w = np.concatenate([tmp_w_lt, tmp_w_et], axis=0)
        tmp_l = np.concatenate([np.zeros_like(tmp_w_lt), np.ones_like(tmp_w_et)], axis=0)

        n_pair = n_et * n_lt
        tot_pair += n_pair
        ret += n_pair * roc_auc_score(tmp_l, tmp_w)
    tot_pair = max(tot_pair, 1e-9)
    ret /= tot_pair
    return ret
    
