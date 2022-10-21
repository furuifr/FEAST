import sys
import os

from pandas.core.reshape.concat import concat
sys.path.append(os.getcwd())
from featureSelectionInMultiParties import featureSelection
import yaml
from preprocessing import getData, sampling
from discretization.dsct import dsct
import random
import numpy as np
import pandas as pd
import warnings;
warnings.filterwarnings(action='once')
warnings.filterwarnings("ignore")
RANDOMSEED = 1
random.seed(RANDOMSEED)

if __name__ == '__main__':
    random.seed(RANDOMSEED)

    with open('multi-party-simulation/physionet_once/physionet_param_unbalance.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    new_col_num = cfg['new_col_num']

    for sub_col_num in range(new_col_num):
        if (sub_col_num+1)%5 != 0:
            continue

        dataset_name = cfg['dataset_name']
        task = cfg['task']
        target = cfg['target']
        # a_col = cfg['a_col']
        # b_col = cfg['b_col']
        # c_col = cfg['c_col']
        # d_col = cfg['d_col']
        a_col = [15,10,17,39,21,3,12,9,6,0,8,5,14,7,29,25,32,30,24,16,31] 
        b_col = [28,22,27,26,38,40,19,4,35,20,13,33,23,37,36,2,18,34,1,11]
        # a_col = np.arange(0, 20).tolist()
        # b_col = np.arange(20, 41).tolist()
        step = cfg['step']
        dsct_method = cfg['dsct_method']
        dsct_num = cfg['dsct_num']
        sampling_method = cfg['sampling_method']

        data = getData(dataset_name, task=task)
        target = data[target]

        if task:
            dataset_name = dataset_name+str(task)
        cols = len(data.columns)-1
        index = list(range(0,cols))
        data_cut, target_cut = sampling(data.iloc[:, index].copy(), target.copy(), sampling_method, dsct_num, step, RANDOMSEED)

        all_columns = list(data.iloc[:, :-1].columns)

        col_set = set(a_col+b_col)

        data_a = pd.DataFrame()
        data_b = pd.DataFrame()
        data_a_cut = pd.DataFrame()
        data_b_cut = pd.DataFrame()

        for i in range(len(a_col)):
            data_a[data.columns.tolist()[a_col[i]]] = data.iloc[:, a_col[i]]
            data_a_cut[data_cut.columns.tolist()[a_col[i]]] = data_cut.iloc[:, a_col[i]]
        
        for i in range(len(b_col)):
            data_b[data.columns.tolist()[b_col[i]]] = data.iloc[:, b_col[i]]
            data_b_cut[data_cut.columns.tolist()[b_col[i]]] = data_cut.iloc[:, b_col[i]]

        data_party = [data_a,data_b]
        data_party_cut = [data_a_cut,data_b_cut]
        
        L = len(data_party)

        data_party_equi = []
        for i in range(L):
            data_party_equi.append(dsct(dsct_method, data_party_cut[i].copy(), dsct_num))

        method = '_FEAST_'
        subdir = 'phys/FEAST/del0.1/1/unbalance/true'
        isExists = os.path.exists(subdir)
        if not isExists:
            os.makedirs(subdir)

        featureSelection(sub_col_num+1, L, data_party, data_party_equi, target_cut, target, method, dataset_name, dsct_method, step, all_columns, RANDOMSEED, subdir)