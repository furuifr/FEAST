import sys
import os

from pandas.core.reshape.concat import concat
sys.path.append(os.getcwd())
from featureSelectionInMultiParties import featureSelection
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

if __name__ == '__main__':
    random.seed(RANDOMSEED)

    with open('multi-party-simulation/census_once/census_param.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    new_col_num = cfg['new_col_num']

    dataset_name = cfg['dataset_name']
    task = cfg['task']
    target = cfg['target']
    a_col = cfg['a_col']
    b_col = cfg['b_col']
    c_col = cfg['c_col']
    d_col = cfg['d_col']
    step = cfg['step']
    dsct_method = cfg['dsct_method']
    dsct_num = cfg['dsct_num']
    sampling_method = cfg['sampling_method']
    
    data = getData(dataset_name, task=task)
    target = data[target]
    data = data.iloc[:, :-1]

    acc_res_fl_ratio_logi = []
    acc_res_fl_ratio_svc = []
    acc_res_fl_ratio_rfc = []
    acc_res_fl_ratio_xgbc = []
    acc_res_fl_ratio_dnn = []

    acc_res_fl_ratio_logi.append(clf_cv('logi', data.copy(), target, RANDOMSEED))
    # acc_res_fl_ratio_svc.append(clf_cv('svc', data.copy(), target, RANDOMSEED))
    acc_res_fl_ratio_rfc.append(clf_cv('rfc', data.copy(), target, RANDOMSEED))
    acc_res_fl_ratio_xgbc.append(clf_cv('xgbc', data.copy(), target, RANDOMSEED))
    acc_res_fl_ratio_dnn.append(clf_cv('dnn', data.copy(), target, RANDOMSEED))

    print(acc_res_fl_ratio_logi)
    print(acc_res_fl_ratio_svc)
    print(acc_res_fl_ratio_rfc)
    print(acc_res_fl_ratio_xgbc)
    print(acc_res_fl_ratio_dnn)


    subdir = "census/ALL/1"
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