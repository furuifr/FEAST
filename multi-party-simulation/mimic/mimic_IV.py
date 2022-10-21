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

def CalcIV(Xvar, Yvar): 
    E = 0.5
    N_0 = np.sum(Yvar==0)
    N_1 = np.sum(Yvar==1)
    N_0_group = np.zeros(np.unique(Xvar).shape)
    N_1_group = np.zeros(np.unique(Xvar).shape)
    for i in range(len(np.unique(Xvar))):
        N_0_group[i] = Yvar[(Xvar == np.unique(Xvar)[i]) & (Yvar == 0)].count()
        N_1_group[i] = Yvar[(Xvar == np.unique(Xvar)[i]) & (Yvar == 1)].count()
    iv = np.sum((N_1_group/N_1 - N_0_group/N_0) * np.log(((N_1_group+E)/N_1)/((N_0_group+E)/N_0)))
    return iv

def GetIVList(df, Y):
    ivlist = []
    for col in df.columns:
        iv = CalcIV(df[col], Y)
        ivlist.append(iv)
    return ivlist

if __name__ == '__main__':
    random.seed(RANDOMSEED)

    with open('multi-party-simulation/mimic_once/mimic_param.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    new_col_num = cfg['new_col_num']

    for sub_col_num in range(1, new_col_num+1):
    
        dataset_name = cfg['dataset_name']
        task = cfg['task']
        target = cfg['target']
        a_col = cfg['a_col']
        b_col = cfg['b_col']
        c_col = cfg['c_col']
        d_col = cfg['d_col']
        step = cfg['step']
        # clf_name = cfg['clf_name']
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

        # acc_res_all = xgbc_cv(data.iloc[:, list(col_set)], data[target])

        data_a = data.iloc[:, a_col]
        data_b = data.iloc[:, b_col]
        data_c = data.iloc[:, c_col]
        data_d = data.iloc[:, d_col]
        data_party = [data_a,data_b,data_c,data_d]

        data_a_cut = data_cut.iloc[:, a_col]
        data_b_cut = data_cut.iloc[:, b_col]
        data_c_cut = data_cut.iloc[:, c_col]
        data_d_cut = data_cut.iloc[:, d_col]
        data_party_cut = [data_a_cut,data_b_cut,data_c_cut,data_d_cut]

        L = len(data_party_cut)

        data_party_equi = []
        for i in range(L):
            # data_party_equi.append(chimerge_dsct(data_party[i].copy(), target, data[target]))
            data_party_equi.append(dsct(dsct_method, data_party_cut[i].copy(), dsct_num))
            
        acc_res_fl_ratio_logi = []
        acc_res_fl_ratio_svc = []
        acc_res_fl_ratio_rfc = []
        acc_res_fl_ratio_xgbc = []
        acc_res_fl_ratio_dnn = []
        model = dataset_name+'_'+dsct_method+'_'+str(step)+'_'+str(sub_col_num)
        method = '_IV_'

        filter_scores = [] # 用于每一轮，其他方针对active方的特征MI
        filter_scores_rank = [] # 用于每一轮，其他方针对active方的特征MI排序

        for i in range(L):
            tmp_scores = np.array(GetIVList(data_party_equi[i].copy(), target_cut))
            tmp_rank = np.argsort(tmp_scores)[::-1]
            filter_scores.append(tmp_scores)
            filter_scores_rank.append(tmp_rank)

        filter_other_party_all = []
        filter_other_party_list = [] # 第一个数为个数，第二个数为scores的总和

        for i in range(len(filter_scores)):
            filter_other_party_list.append([0, 0]) # 第一个数为个数，第二个数为scores的总和
            for j in range(len(filter_scores[i])):
                filter_other_party_all.append((i, filter_scores[i][j]))

        filter_other_party_all = sorted(filter_other_party_all, key=lambda x: -x[1])

        for i in range(sub_col_num):
            filter_other_party_list[filter_other_party_all[i][0]][0] += 1
            filter_other_party_list[filter_other_party_all[i][0]][1] += filter_other_party_all[i][1]
        
        for i in range(len(filter_other_party_list)):
            if i == 0:
                data_sum_fl_ratio = data_party[i].iloc[:, filter_scores_rank[i][:filter_other_party_list[i][0]]]
            else:
                data_sum_fl_ratio = pd.concat([data_sum_fl_ratio, data_party[i].iloc[:, filter_scores_rank[i][:filter_other_party_list[i][0]]]], axis=1)

        acc_res_fl_ratio_logi.append(clf_cv('logi', data_sum_fl_ratio.copy(), target, RANDOMSEED))
        # acc_res_fl_ratio_svc.append(clf_cv('svc', data_sum_fl_ratio.copy(), target, RANDOMSEED))
        acc_res_fl_ratio_rfc.append(clf_cv('rfc', data_sum_fl_ratio.copy(), target, RANDOMSEED))
        acc_res_fl_ratio_xgbc.append(clf_cv('xgbc', data_sum_fl_ratio.copy(), target, RANDOMSEED))
        acc_res_fl_ratio_dnn.append(clf_cv('dnn', data_sum_fl_ratio.copy(), target, RANDOMSEED))


        print(list(data_sum_fl_ratio.columns))
        # print(list(map(lambda x: all_columns.index(x), list(data_sum_fl_ratio.columns))))
        # print(acc_res_fl_ratio)

        print(acc_res_fl_ratio_logi)
        print(acc_res_fl_ratio_svc)
        print(acc_res_fl_ratio_rfc)
        print(acc_res_fl_ratio_xgbc)
        print(acc_res_fl_ratio_dnn)

        subdir = "mimic0/IV/1"
        isExists = os.path.exists(subdir)
        if not isExists:
            os.makedirs(subdir)

        with open(subdir+'/logi_'+model+method+'.txt', 'w') as f:
            f.write(str(list(data_sum_fl_ratio.columns)))
            f.write('\r\n')
            f.write(str(list(map(lambda x: all_columns.index(x), list(data_sum_fl_ratio.columns)))))
            f.write('\r\n')
            f.write(str(acc_res_fl_ratio_logi))
            f.write('\r\n')

        # with open(subdir+'/svc_'+model+method+'.txt', 'w') as f:
        #     f.write(str(list(data_sum_fl_ratio.columns)))
        #     f.write('\r\n')
        #     f.write(str(list(map(lambda x: all_columns.index(x), list(data_sum_fl_ratio.columns)))))
        #     f.write('\r\n')
        #     f.write(str(acc_res_fl_ratio_svc))
        #     f.write('\r\n')

        with open(subdir+'/rfc_'+model+method+'.txt', 'w') as f:
            f.write(str(list(data_sum_fl_ratio.columns)))
            f.write('\r\n')
            f.write(str(list(map(lambda x: all_columns.index(x), list(data_sum_fl_ratio.columns)))))
            f.write('\r\n')
            f.write(str(acc_res_fl_ratio_rfc))
            f.write('\r\n')

        with open(subdir+'/xgbc_'+model+method+'.txt', 'w') as f:
            f.write(str(list(data_sum_fl_ratio.columns)))
            f.write('\r\n')
            f.write(str(list(map(lambda x: all_columns.index(x), list(data_sum_fl_ratio.columns)))))
            f.write('\r\n')
            f.write(str(acc_res_fl_ratio_xgbc))
            f.write('\r\n')

        with open(subdir+'/dnn_'+model+method+'.txt', 'w') as f:
            f.write(str(list(data_sum_fl_ratio.columns)))
            f.write('\r\n')
            f.write(str(list(map(lambda x: all_columns.index(x), list(data_sum_fl_ratio.columns)))))
            f.write('\r\n')
            f.write(str(acc_res_fl_ratio_dnn))
            f.write('\r\n')



