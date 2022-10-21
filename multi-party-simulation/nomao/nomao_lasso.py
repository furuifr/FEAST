import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
import yaml
from preprocessing import getData, sampling
from discretization.dsct import dsct
from classification import clf_cv
import random
import warnings;
warnings.filterwarnings(action='once')
warnings.filterwarnings("ignore")
RANDOMSEED = 1
random.seed(RANDOMSEED)
from sklearn.linear_model import Lasso


if __name__ == '__main__':
    random.seed(RANDOMSEED)

    with open('multi-party-simulation/nomao_once/nomao_param.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    new_col_num = cfg['new_col_num']
    dataset_name = cfg['dataset_name']
    task = cfg['task']
    target = cfg['target']
    a_col = np.arange(1, 34).tolist()
    b_col = np.arange(30, 64).tolist()
    c_col = np.arange(60, 94).tolist()
    d_col = np.arange(90, 119).tolist()+np.arange(1, 5).tolist()
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
    data_c = data.iloc[:, c_col]
    data_d = data.iloc[:, d_col]

    data_party = [data_a,data_b,data_c,data_d]

    L = len(data_party)

    data_party_equi = []
    X = pd.DataFrame()
    for i in range(L):
        data_party[i] = data_party[i].rename(lambda x: x+'_'+str(i), axis=1)
        X = pd.concat([X, data_party[i]], axis=1)

    all_columns = list(X.columns)

    lamb = 0.000005
    lasso_reg = Lasso(alpha=lamb)
    lasso_reg.fit(X, target)
    print(lasso_reg.intercept_, lasso_reg.coef_)
    feature_ranking = np.argsort(-np.array(abs(lasso_reg.coef_)))

    for sub_col_num in range(1, new_col_num+1):
            
        acc_res_fl_ratio_logi = []
        acc_res_fl_ratio_svc = []
        acc_res_fl_ratio_rfc = []
        acc_res_fl_ratio_xgbc = []
        acc_res_fl_ratio_dnn = []
        model = dataset_name+'_'+dsct_method+'_'+str(step)+'_'+str(sub_col_num)
        method = '_Lasso_'

        sub_data = X.iloc[:, feature_ranking[:sub_col_num]]

        acc_res_fl_ratio_logi.append(clf_cv('logi', sub_data.copy(), target, RANDOMSEED))
        acc_res_fl_ratio_svc.append(clf_cv('svc', sub_data.copy(), target, RANDOMSEED))
        acc_res_fl_ratio_rfc.append(clf_cv('rfc', sub_data.copy(), target, RANDOMSEED))
        acc_res_fl_ratio_xgbc.append(clf_cv('xgbc', sub_data.copy(), target, RANDOMSEED))
        acc_res_fl_ratio_dnn.append(clf_cv('dnn', sub_data.copy(), target, RANDOMSEED))

        print(list(sub_data.columns))

        print(acc_res_fl_ratio_logi)
        print(acc_res_fl_ratio_svc)
        print(acc_res_fl_ratio_rfc)
        print(acc_res_fl_ratio_xgbc)
        print(acc_res_fl_ratio_dnn)

        subdir = "nomao/Lasso/1"
        isExists = os.path.exists(subdir)
        if not isExists:
            os.makedirs(subdir)

        with open(subdir+'/logi_'+model+method+'.txt', 'w') as f:
            f.write(str(list(sub_data.columns)))
            f.write('\r\n')
            f.write(str(list(map(lambda x: all_columns.index(x), list(sub_data.columns)))))
            f.write('\r\n')
            f.write(str(acc_res_fl_ratio_logi))
            f.write('\r\n')

        with open(subdir+'/svc_'+model+method+'.txt', 'w') as f:
            f.write(str(list(sub_data.columns)))
            f.write('\r\n')
            f.write(str(list(map(lambda x: all_columns.index(x), list(sub_data.columns)))))
            f.write('\r\n')
            f.write(str(acc_res_fl_ratio_svc))
            f.write('\r\n')

        with open(subdir+'/rfc_'+model+method+'.txt', 'w') as f:
            f.write(str(list(sub_data.columns)))
            f.write('\r\n')
            f.write(str(list(map(lambda x: all_columns.index(x), list(sub_data.columns)))))
            f.write('\r\n')
            f.write(str(acc_res_fl_ratio_rfc))
            f.write('\r\n')

        with open(subdir+'/xgbc_'+model+method+'.txt', 'w') as f:
            f.write(str(list(sub_data.columns)))
            f.write('\r\n')
            f.write(str(list(map(lambda x: all_columns.index(x), list(sub_data.columns)))))
            f.write('\r\n')
            f.write(str(acc_res_fl_ratio_xgbc))
            f.write('\r\n')

        with open(subdir+'/dnn_'+model+method+'.txt', 'w') as f:
            f.write(str(list(sub_data.columns)))
            f.write('\r\n')
            f.write(str(list(map(lambda x: all_columns.index(x), list(sub_data.columns)))))
            f.write('\r\n')
            f.write(str(acc_res_fl_ratio_dnn))
            f.write('\r\n')

