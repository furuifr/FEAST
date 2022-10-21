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
import warnings;
warnings.filterwarnings(action='once')
warnings.filterwarnings("ignore")
RANDOMSEED = 2
random.seed(RANDOMSEED)

if __name__ == '__main__':
    random.seed(RANDOMSEED)

    with open('multi-party-simulation/physionet_once/physionet_param_overlapping.yaml', 'r') as f:
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
        a_col = np.arange(0, 35).tolist()
        b_col = np.arange(5, 41).tolist()
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

        data_a = data.iloc[:, a_col]
        data_b = data.iloc[:, b_col]


        data_a_cut = data_cut.iloc[:, a_col]
        data_b_cut = data_cut.iloc[:, b_col]

        data_party = [data_a,data_b]
        data_party_cut = [data_a_cut,data_b_cut]
        
        L = len(data_party)

        data_party_equi = []
        for i in range(L):
            data_party_equi.append(dsct(dsct_method, data_party_cut[i].copy(), dsct_num))

        method = '_FEAST_'
        subdir = 'phys/FEAST/del0.1/2/OverlappingTest30'
        isExists = os.path.exists(subdir)
        if not isExists:
            os.makedirs(subdir)

        featureSelection(sub_col_num+1, L, data_party, data_party_equi, target_cut, target, method, dataset_name, dsct_method, step, all_columns, RANDOMSEED, subdir)