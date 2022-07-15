import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
import yaml
from preprocessing import getData
from discretization.dsct import dsct
from filter.mutual_info_single import calc_cond_MI, calc_MI
from featureSelectionInSingleParty import featureSelection
from classification import clf_cv, rfc_cv
from pprint import pprint
import random
import warnings;
warnings.filterwarnings(action='once')
warnings.filterwarnings("ignore")
RANDOMSEED = 1
random.seed(RANDOMSEED)


if __name__ == '__main__':
    random.seed(RANDOMSEED)

    with open('single-party/mimic/mimic_param.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    dataset_name = cfg['dataset_name']
    task = cfg['task']
    target = cfg['target']
    new_col_num = cfg['new_col_num']
    a_col = cfg['a_col']
    b_col = cfg['b_col']
    c_col = cfg['c_col']
    d_col = cfg['d_col']
    step = cfg['step']
    # clf_name = cfg['clf_name']
    dsct_method = cfg['dsct_method']
    dsct_num = cfg['dsct_num']

    data = getData(dataset_name)
    # if task:
    #     dataset_name = dataset_name+str(task)

    target = data[target]

    cols = len(data.columns)-1
    index = list(range(0,cols))

    all_columns = list(data.iloc[:, :-1].columns)

    # a_col = [10,21,17,32,16,19,40,27]
    # random.shuffle(index)
    # a_col = index[:8]
    # random.shuffle(index)
    # b_col = index[:17]
    # random.shuffle(index)
    # c_col = index[:15]
    # random.shuffle(index)
    # d_col = index[:13]

    col_set = set(a_col+b_col+c_col+d_col)
    col_list = list(col_set)

    data = data.iloc[:, col_list]
    data_dsct = dsct(dsct_method, data.copy(), dsct_num)

    method = '_FEAST_'
    subdir = '1'
    isExists = os.path.exists(subdir)
    if not isExists:
        os.makedirs(subdir)

    featureSelection(new_col_num, data, data_dsct, target, method, dataset_name, dsct_method, all_columns, RANDOMSEED, subdir)