import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
import yaml
from preprocessing import getData, sampling
from discretization.dsct import dsct
from classification import clf_cv
from filter.mutual_info_multi_del import calc_MI
import random
import warnings;
warnings.filterwarnings(action='once')
warnings.filterwarnings("ignore")
RANDOMSEED = 2
random.seed(RANDOMSEED)

if __name__ == '__main__':
    random.seed(RANDOMSEED)

    with open('multi-party-simulation/physionet_once/physionet_param.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    new_col_num = cfg['new_col_num']
    dataset_name = cfg['dataset_name']
    task = cfg['task']
    target = cfg['target']
    a_col = np.arange(0, 35).tolist()
    b_col = np.arange(5, 41).tolist()
    step = cfg['step']
    # clf_name = cfg['clf_name']
    dsct_method = cfg['dsct_method']
    dsct_num = cfg['dsct_num']
    sampling_method = cfg['sampling_method']

    data = getData(dataset_name, task=task)
    target = data[target]

    if task:
        dataset_name = dataset_name+str(task)

    data_a = data.iloc[:, a_col]
    data_b = data.iloc[:, b_col]

    data_party = [data_a,data_b]
    L = len(data_party)

    X = pd.DataFrame()
    for i in range(L):
        data_party[i] = data_party[i].rename(lambda x: x+'_'+str(i), axis=1)
        X = pd.concat([X, data_party[i]], axis=1)

    acc_res_fl_ratio_logi = []
    acc_res_fl_ratio_svc = []
    acc_res_fl_ratio_rfc = []
    acc_res_fl_ratio_xgbc = []
    acc_res_fl_ratio_dnn = []

    acc_res_fl_ratio_logi.append(clf_cv('logi', X.copy(), target, RANDOMSEED))
    # acc_res_fl_ratio_svc.append(clf_cv('svc', data.copy(), target, RANDOMSEED))
    acc_res_fl_ratio_rfc.append(clf_cv('rfc', X.copy(), target, RANDOMSEED))
    acc_res_fl_ratio_xgbc.append(clf_cv('xgbc', X.copy(), target, RANDOMSEED))
    acc_res_fl_ratio_dnn.append(clf_cv('dnn', X.copy(), target, RANDOMSEED))

    print(acc_res_fl_ratio_logi)
    print(acc_res_fl_ratio_svc)
    print(acc_res_fl_ratio_rfc)
    print(acc_res_fl_ratio_xgbc)
    print(acc_res_fl_ratio_dnn)

    subdir = "phys/ALL_overlapping/2/40"
    isExists = os.path.exists(subdir)
    if not isExists:
        os.makedirs(subdir)


    with open(subdir+'/logi.txt', 'w') as f:
        f.write(str(acc_res_fl_ratio_logi))
        f.write('\r\n')

    # with open(subdir+'/svc.txt', 'w') as f:
    #     f.write(str(acc_res_fl_ratio_svc))
    #     f.write('\r\n')

    with open(subdir+'/rfc.txt', 'w') as f:
        f.write(str(acc_res_fl_ratio_rfc))
        f.write('\r\n')

    with open(subdir+'/xgbc.txt', 'w') as f:
        f.write(str(acc_res_fl_ratio_xgbc))
        f.write('\r\n')

    with open(subdir+'/dnn.txt', 'w') as f:
        f.write(str(acc_res_fl_ratio_dnn))
        f.write('\r\n')