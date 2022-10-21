import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
import yaml
from preprocessing import getData
from discretization.dsct import dsct
from filter.mutual_info_multi_del import calc_MI, getCrosstab, calc_cond_MI
from classification import clf_cv
from pprint import pprint
import random
import warnings;
warnings.filterwarnings(action='once')
warnings.filterwarnings("ignore")
import time

# featureSelection_ori
def featureSelection_ori(new_col_num, L, data_party, data_party_equi, target, method, dataset_name, dsct_method, step, all_columns):
    cross_feature = [None for _ in range(L)]
    data_sum_fl_ratio = data_party[0]
    acc_res_fl_ratio_logi = []
    acc_res_fl_ratio_svc = []
    acc_res_fl_ratio_rfc = []
    acc_res_fl_ratio_xgbc = []
    cond_MI = [[] for _ in range(L)] # 用于每一轮，其他方针对active方的特征MI
    cond_MI_rank = [[] for _ in range(L)] # 用于每一轮，其他方针对active方的特征MI排序

    model = dataset_name+'_'+dsct_method+'_'+str(step)+'_'+str(new_col_num)

    # 并行，只考虑已有特征
    for i in range(L-1):
        if new_col_num==0:
            break
        if i==0:
            cross_feature[i] = getCrosstab(data_party_equi[i], target, step, model+method+str(i))
        else:
            cross_feature[i] = pd.concat([cross_feature[i-1], getCrosstab(data_party_equi[i], target, step, model+method+str(i))], axis=1)

        for j in range(i+1, L):
            tmp_cond_MI, tmp_cond_MI_rank = calc_cond_MI(cross_feature[i], data_party_equi[j], target, method)
            cond_MI[i].append(tmp_cond_MI)
            cond_MI_rank[i].append(tmp_cond_MI_rank)

        cond_MI_other_party_all = []
        cond_MI_other_party_list = [] # 第一个数为个数，第二个数为MI的总和

        for j in range(len(cond_MI[i])):
            cond_MI_other_party_list.append([0, 0]) # 第一个数为个数，第二个数为MI的总和
            for k in range(len(cond_MI[i][j])):
                cond_MI_other_party_all.append((j, cond_MI[i][j][k]))

        cond_MI_other_party_all = sorted(cond_MI_other_party_all, key=lambda x: -x[1])

        # 这里先选总和最大的，之后再调整
        for j in range(new_col_num):
            cond_MI_other_party_list[cond_MI_other_party_all[j][0]][0] += 1
            cond_MI_other_party_list[cond_MI_other_party_all[j][0]][1] += cond_MI_other_party_all[j][1]

        for j in range(len(cond_MI_other_party_list)):
            if cond_MI_other_party_list[j][0] == 0:
                cond_MI_other_party_list[j][0] = 10000
            cond_MI_other_party_list[j][1] /= cond_MI_other_party_list[j][0]

        cond_MI_other_party_arr = np.array(cond_MI_other_party_list)
        cond_MI_other_party_rank = np.argsort(-cond_MI_other_party_arr, axis=0) # 获取MI总和的排序,axis=0为按列排序，即数组间纵向比较排序,axis=1为按行排序，即数组内横向比较排序
        cond_MI_other_party_rank = list(map(lambda x:x[1], cond_MI_other_party_rank))

        next_active = cond_MI_other_party_rank[0] # 下一个发起方
        next_feature_num = cond_MI_other_party_list[next_active][0] # 下一个发起方的特征个数
        data_party.insert(i+1, data_party.pop(next_active+i+1))
        data_party_equi.insert(i+1, data_party_equi.pop(next_active+i+1))

        data_party[i+1], data_party_equi[i+1] = data_party[i+1].iloc[:, cond_MI_rank[i][next_active][:next_feature_num]], data_party_equi[i+1].iloc[:, cond_MI_rank[i][next_active][:next_feature_num]]

        data_sum_fl_ratio = pd.concat([data_sum_fl_ratio, data_party[i+1]], axis=1)
        # acc_res_fl_ratio.append(clf_cv(clf_name, data_sum_fl_ratio.copy().T.drop_duplicates().T, target))
        acc_res_fl_ratio_logi.append(clf_cv('logi', data_sum_fl_ratio.copy().T.drop_duplicates().T, target))
        acc_res_fl_ratio_svc.append(clf_cv('svc', data_sum_fl_ratio.copy().T.drop_duplicates().T, target))
        acc_res_fl_ratio_rfc.append(clf_cv('rfc', data_sum_fl_ratio.copy().T.drop_duplicates().T, target))
        acc_res_fl_ratio_xgbc.append(clf_cv('xgbc', data_sum_fl_ratio.copy().T.drop_duplicates().T, target))

        new_col_num -= next_feature_num # 删除已选的个数

    print(list(data_sum_fl_ratio.columns))
    # print(list(map(lambda x: all_columns.index(x), list(data_sum_fl_ratio.columns))))
    # print(acc_res_fl_ratio)

    print(acc_res_fl_ratio_logi)
    print(acc_res_fl_ratio_svc)
    print(acc_res_fl_ratio_rfc)
    print(acc_res_fl_ratio_xgbc)

    with open('logi_'+model+method+'.txt', 'w') as f:
        f.write(str(list(data_sum_fl_ratio.columns)))
        f.write('\r\n')
        f.write(str(list(map(lambda x: all_columns.index(x), list(data_sum_fl_ratio.columns)))))
        f.write('\r\n')
        f.write(str(acc_res_fl_ratio_logi))
        f.write('\r\n')

    with open('svc_'+model+method+'.txt', 'w') as f:
        f.write(str(list(data_sum_fl_ratio.columns)))
        f.write('\r\n')
        f.write(str(list(map(lambda x: all_columns.index(x), list(data_sum_fl_ratio.columns)))))
        f.write('\r\n')
        f.write(str(acc_res_fl_ratio_svc))
        f.write('\r\n')

    with open('rfc_'+model+method+'.txt', 'w') as f:
        f.write(str(list(data_sum_fl_ratio.columns)))
        f.write('\r\n')
        f.write(str(list(map(lambda x: all_columns.index(x), list(data_sum_fl_ratio.columns)))))
        f.write('\r\n')
        f.write(str(acc_res_fl_ratio_rfc))
        f.write('\r\n')

    with open('xgbc_'+model+method+'.txt', 'w') as f:
        f.write(str(list(data_sum_fl_ratio.columns)))
        f.write('\r\n')
        f.write(str(list(map(lambda x: all_columns.index(x), list(data_sum_fl_ratio.columns)))))
        f.write('\r\n')
        f.write(str(acc_res_fl_ratio_xgbc))
        f.write('\r\n')

# featureSelection_del 默认把第一家全选
def featureSelection_del(new_col_num, L, data_party, data_party_equi, target, target_pred, method, dataset_name, dsct_method, step, all_columns):
    cross_feature = [None for _ in range(L)]
    data_sum_fl_ratio = data_party[0]
    acc_res_fl_ratio_logi = []
    acc_res_fl_ratio_svc = []
    acc_res_fl_ratio_rfc = []
    acc_res_fl_ratio_xgbc = []
    cond_MI = [[] for _ in range(L)] # 用于每一轮，其他方针对active方的特征MI
    cond_MI_rank = [[] for _ in range(L)] # 用于每一轮，其他方针对active方的特征MI排序

    model = dataset_name+'_'+dsct_method+'_'+str(step)+'_'+str(new_col_num)

    # 并行，只考虑已有特征
    for i in range(L-1):
        if new_col_num==0:
            break
        if i==0:
            cross_feature[i] = getCrosstab(data_party_equi[i], target, step, model+method+str(i))
        else:
            cross_feature[i] = pd.concat([cross_feature[i-1], getCrosstab(data_party_equi[i], target, step, model+method+str(i))], axis=1)

        for j in range(i+1, L):
            tmp_cond_MI, tmp_cond_MI_rank = calc_cond_MI(cross_feature[i], data_party_equi[j], target, method)
            cond_MI[i].append(tmp_cond_MI)
            cond_MI_rank[i].append(tmp_cond_MI_rank)

        cond_MI_other_party_all = []
        cond_MI_other_party_list = [] # 第一个数为个数，第二个数为MI的总和
        other_party_delete_list = []

        for j in range(len(cond_MI[i])):
            cond_MI_other_party_list.append([0, 0]) # 第一个数为个数，第二个数为MI的总和
            other_party_delete_list.append(0) # 要删除的个数
            for k in range(len(cond_MI[i][j])):
                cond_MI_other_party_all.append((j, cond_MI[i][j][k]))

        cond_MI_other_party_all = sorted(cond_MI_other_party_all, key=lambda x: -x[1])
        
        delete_num = int((len(cond_MI_other_party_all)-new_col_num) * 0.2)
        # 遍历后delete_num个features
        for j in range(len(cond_MI_other_party_all)-delete_num, len(cond_MI_other_party_all)):
            other_party_delete_list[cond_MI_other_party_all[j][0]] += 1

        # 这里先选总和最大的，之后再调整
        for j in range(new_col_num):
            cond_MI_other_party_list[cond_MI_other_party_all[j][0]][0] += 1
            cond_MI_other_party_list[cond_MI_other_party_all[j][0]][1] += cond_MI_other_party_all[j][1]

        for j in range(len(cond_MI_other_party_list)):
            if cond_MI_other_party_list[j][0] == 0:
                cond_MI_other_party_list[j][0] = 10000
            cond_MI_other_party_list[j][1] /= cond_MI_other_party_list[j][0]

        cond_MI_other_party_arr = np.array(cond_MI_other_party_list)
        cond_MI_other_party_rank = np.argsort(-cond_MI_other_party_arr, axis=0) # 获取MI总和的排序,axis=0为按列排序，即数组间纵向比较排序,axis=1为按行排序，即数组内横向比较排序
        cond_MI_other_party_rank = list(map(lambda x:x[1], cond_MI_other_party_rank))

        next_active = cond_MI_other_party_rank[0] # 下一个发起方
        next_feature_num = cond_MI_other_party_list[next_active][0] # 下一个发起方的特征个数

        # data_party[i+1], data_party_equi[i+1] = data_party[i+1].iloc[:, cond_MI_rank[i][next_active][:next_feature_num]], data_party_equi[i+1].iloc[:, cond_MI_rank[i][next_active][:next_feature_num]]

        for j in range(i+1, L):
            if j == next_active+i+1:
                data_party[j], data_party_equi[j] = data_party[j].iloc[:, cond_MI_rank[i][j-1-i][:next_feature_num]], data_party_equi[j].iloc[:, cond_MI_rank[i][j-1-i][:next_feature_num]]
            else:
                if other_party_delete_list[j-1-i] != 0:
                    data_party[j], data_party_equi[j] = data_party[j].iloc[:, cond_MI_rank[i][j-1-i][:-other_party_delete_list[j-1-i]]], data_party_equi[j].iloc[:, cond_MI_rank[i][j-1-i][:-other_party_delete_list[j-1-i]]]

        data_party.insert(i+1, data_party.pop(next_active+i+1))
        data_party_equi.insert(i+1, data_party_equi.pop(next_active+i+1))
        
        data_sum_fl_ratio = pd.concat([data_sum_fl_ratio, data_party[i+1]], axis=1)
        # acc_res_fl_ratio.append(clf_cv(clf_name, data_sum_fl_ratio.copy().T.drop_duplicates().T, target))
        acc_res_fl_ratio_logi.append(clf_cv('logi', data_sum_fl_ratio.copy().T.drop_duplicates().T, target_pred))
        acc_res_fl_ratio_svc.append(clf_cv('svc', data_sum_fl_ratio.copy().T.drop_duplicates().T, target_pred))
        acc_res_fl_ratio_rfc.append(clf_cv('rfc', data_sum_fl_ratio.copy().T.drop_duplicates().T, target_pred))
        acc_res_fl_ratio_xgbc.append(clf_cv('xgbc', data_sum_fl_ratio.copy().T.drop_duplicates().T, target_pred))

        new_col_num -= next_feature_num # 删除已选的个数

    print(list(data_sum_fl_ratio.columns))
    # print(list(map(lambda x: all_columns.index(x), list(data_sum_fl_ratio.columns))))
    # print(acc_res_fl_ratio)

    print(acc_res_fl_ratio_logi)
    print(acc_res_fl_ratio_svc)
    print(acc_res_fl_ratio_rfc)
    print(acc_res_fl_ratio_xgbc)

    with open('logi_'+model+method+'.txt', 'w') as f:
        f.write(str(list(data_sum_fl_ratio.columns)))
        f.write('\r\n')
        f.write(str(list(map(lambda x: all_columns.index(x), list(data_sum_fl_ratio.columns)))))
        f.write('\r\n')
        f.write(str(acc_res_fl_ratio_logi))
        f.write('\r\n')

    with open('svc_'+model+method+'.txt', 'w') as f:
        f.write(str(list(data_sum_fl_ratio.columns)))
        f.write('\r\n')
        f.write(str(list(map(lambda x: all_columns.index(x), list(data_sum_fl_ratio.columns)))))
        f.write('\r\n')
        f.write(str(acc_res_fl_ratio_svc))
        f.write('\r\n')

    with open('rfc_'+model+method+'.txt', 'w') as f:
        f.write(str(list(data_sum_fl_ratio.columns)))
        f.write('\r\n')
        f.write(str(list(map(lambda x: all_columns.index(x), list(data_sum_fl_ratio.columns)))))
        f.write('\r\n')
        f.write(str(acc_res_fl_ratio_rfc))
        f.write('\r\n')

    with open('xgbc_'+model+method+'.txt', 'w') as f:
        f.write(str(list(data_sum_fl_ratio.columns)))
        f.write('\r\n')
        f.write(str(list(map(lambda x: all_columns.index(x), list(data_sum_fl_ratio.columns)))))
        f.write('\r\n')
        f.write(str(acc_res_fl_ratio_xgbc))
        f.write('\r\n')

# featureSelection_selectall 从第一家就开始选择，不删除tail features
def featureSelection_selectall(new_col_num, L, data_party, data_party_equi, target, target_pred, method, dataset_name, dsct_method, step, all_columns, RANDOMSEED):
    cross_feature = [None for _ in range(L)]
    data_sum_fl_ratio = None
    acc_res_fl_ratio_logi = []
    acc_res_fl_ratio_svc = []
    acc_res_fl_ratio_rfc = []
    acc_res_fl_ratio_xgbc = []
    cond_MI = [[] for _ in range(L)] # 用于每一轮，其他方针对active方的特征MI
    cond_MI_rank = [[] for _ in range(L)] # 用于每一轮，其他方针对active方的特征MI排序

    model = dataset_name+'_'+dsct_method+'_'+str(step)+'_'+str(new_col_num)

    # 并行，只考虑已有特征
    for i in range(L):

        if new_col_num==0:
            break

        for j in range(i, L):
            if i == 0:
                tmp_cond_MI, tmp_cond_MI_rank = calc_MI(data_party_equi[j], target)
            else:
                tmp_cond_MI, tmp_cond_MI_rank = calc_cond_MI(cross_feature[i-1], data_party_equi[j], target, method)
            cond_MI[i].append(tmp_cond_MI)
            cond_MI_rank[i].append(tmp_cond_MI_rank)

        cond_MI_other_party_all = []
        cond_MI_other_party_list = [] # 第一个数为个数，第二个数为MI的总和
        # other_party_delete_list = []

        for j in range(len(cond_MI[i])):
            cond_MI_other_party_list.append([0, 0]) # 第一个数为个数，第二个数为MI的总和
            # other_party_delete_list.append(0) # 要删除的个数
            for k in range(len(cond_MI[i][j])):
                cond_MI_other_party_all.append((j, cond_MI[i][j][k]))

        cond_MI_other_party_all = sorted(cond_MI_other_party_all, key=lambda x: -x[1])
        
        # delete_num = int((len(cond_MI_other_party_all)-new_col_num) * 0.05)
        # 遍历后delete_num个features
        # for j in range(len(cond_MI_other_party_all)-delete_num, len(cond_MI_other_party_all)):
        #     other_party_delete_list[cond_MI_other_party_all[j][0]] += 1

        # 这里先选总和最大的，之后再调整
        for j in range(new_col_num):
            if j >= len(cond_MI_other_party_all):
                break
            cond_MI_other_party_list[cond_MI_other_party_all[j][0]][0] += 1
            cond_MI_other_party_list[cond_MI_other_party_all[j][0]][1] += cond_MI_other_party_all[j][1]

        for j in range(len(cond_MI_other_party_list)):
            if cond_MI_other_party_list[j][0] == 0:
                cond_MI_other_party_list[j][0] = 10000
            cond_MI_other_party_list[j][1] /= cond_MI_other_party_list[j][0]

        cond_MI_other_party_arr = np.array(cond_MI_other_party_list)
        cond_MI_other_party_rank = np.argsort(-cond_MI_other_party_arr, axis=0) # 获取MI总和的排序,axis=0为按列排序，即数组间纵向比较排序,axis=1为按行排序，即数组内横向比较排序
        cond_MI_other_party_rank = list(map(lambda x:x[1], cond_MI_other_party_rank))

        next_active = cond_MI_other_party_rank[0] # 下一个发起方
        next_feature_num = cond_MI_other_party_list[next_active][0] # 下一个发起方的特征个数

        # data_party[i+1], data_party_equi[i+1] = data_party[i+1].iloc[:, cond_MI_rank[i][next_active][:next_feature_num]], data_party_equi[i+1].iloc[:, cond_MI_rank[i][next_active][:next_feature_num]]

        for j in range(i, L):
            if j == next_active+i:
                data_party[j], data_party_equi[j] = data_party[j].iloc[:, cond_MI_rank[i][j-i][:next_feature_num]], data_party_equi[j].iloc[:, cond_MI_rank[i][j-i][:next_feature_num]]
            # else:
            #     if other_party_delete_list[j-i] != 0:
            #         data_party[j], data_party_equi[j] = data_party[j].iloc[:, cond_MI_rank[i][j-i][:-other_party_delete_list[j-i]]], data_party_equi[j].iloc[:, cond_MI_rank[i][j-i][:-other_party_delete_list[j-i]]]

        data_party.insert(i, data_party.pop(next_active+i))
        data_party_equi.insert(i, data_party_equi.pop(next_active+i))

        if i==0:
            cross_feature[i] = getCrosstab(data_party_equi[i], target, step, model+method+str(i))
        else:
            cross_feature[i] = pd.concat([cross_feature[i-1], getCrosstab(data_party_equi[i], target, step, model+method+str(i))], axis=1)
        
        if i==0:
            data_sum_fl_ratio = data_party[i]
        else:
            data_sum_fl_ratio = pd.concat([data_sum_fl_ratio, data_party[i]], axis=1)
            
        # acc_res_fl_ratio.append(clf_cv(clf_name, data_sum_fl_ratio.copy().T.drop_duplicates().T, target))
        acc_res_fl_ratio_logi.append(clf_cv('logi', data_sum_fl_ratio.copy().T.drop_duplicates().T, target_pred, RANDOMSEED))
        acc_res_fl_ratio_svc.append(clf_cv('svc', data_sum_fl_ratio.copy().T.drop_duplicates().T, target_pred, RANDOMSEED))
        acc_res_fl_ratio_rfc.append(clf_cv('rfc', data_sum_fl_ratio.copy().T.drop_duplicates().T, target_pred, RANDOMSEED))
        acc_res_fl_ratio_xgbc.append(clf_cv('xgbc', data_sum_fl_ratio.copy().T.drop_duplicates().T, target_pred, RANDOMSEED))

        new_col_num -= next_feature_num # 删除已选的个数

    print(list(data_sum_fl_ratio.columns))
    # print(list(map(lambda x: all_columns.index(x), list(data_sum_fl_ratio.columns))))
    # print(acc_res_fl_ratio)

    print(acc_res_fl_ratio_logi)
    print(acc_res_fl_ratio_svc)
    print(acc_res_fl_ratio_rfc)
    print(acc_res_fl_ratio_xgbc)

    with open('logi_'+model+method+'.txt', 'w') as f:
        f.write(str(list(data_sum_fl_ratio.columns)))
        f.write('\r\n')
        f.write(str(list(map(lambda x: all_columns.index(x), list(data_sum_fl_ratio.columns)))))
        f.write('\r\n')
        f.write(str(acc_res_fl_ratio_logi))
        f.write('\r\n')

    with open('svc_'+model+method+'.txt', 'w') as f:
        f.write(str(list(data_sum_fl_ratio.columns)))
        f.write('\r\n')
        f.write(str(list(map(lambda x: all_columns.index(x), list(data_sum_fl_ratio.columns)))))
        f.write('\r\n')
        f.write(str(acc_res_fl_ratio_svc))
        f.write('\r\n')

    with open('rfc_'+model+method+'.txt', 'w') as f:
        f.write(str(list(data_sum_fl_ratio.columns)))
        f.write('\r\n')
        f.write(str(list(map(lambda x: all_columns.index(x), list(data_sum_fl_ratio.columns)))))
        f.write('\r\n')
        f.write(str(acc_res_fl_ratio_rfc))
        f.write('\r\n')

    with open('xgbc_'+model+method+'.txt', 'w') as f:
        f.write(str(list(data_sum_fl_ratio.columns)))
        f.write('\r\n')
        f.write(str(list(map(lambda x: all_columns.index(x), list(data_sum_fl_ratio.columns)))))
        f.write('\r\n')
        f.write(str(acc_res_fl_ratio_xgbc))
        f.write('\r\n')

# featureSelection_del_selectall 从第一家就开始选择，删除tail features
def featureSelection(new_col_num, L, data_party, data_party_equi, target, target_pred, method, dataset_name, dsct_method, step, all_columns, RANDOMSEED, subdir):
    cross_feature = [None for _ in range(L)]
    data_sum_fl_ratio = None
    acc_res_fl_ratio_logi = []
    acc_res_fl_ratio_svc = []
    acc_res_fl_ratio_rfc = []
    acc_res_fl_ratio_xgbc = []
    acc_res_fl_ratio_dnn = []
    cond_MI = [[] for _ in range(L)] # 用于每一轮，其他方针对active方的特征MI
    cond_MI_rank = [[] for _ in range(L)] # 用于每一轮，其他方针对active方的特征MI排序
    sumTime = 0

    model = dataset_name+'_'+dsct_method+'_'+str(step)+'_'+str(new_col_num)

    # 并行，只考虑已有特征
    for i in range(L):

        timeList = []

        if new_col_num==0:
            break

        for j in range(i, L):
            start = time.time()
            if i == 0:
                tmp_cond_MI, tmp_cond_MI_rank = calc_MI(data_party_equi[j], target)
            else:
                tmp_cond_MI, tmp_cond_MI_rank = calc_cond_MI(cross_feature[i-1], data_party_equi[j], target, method)
            cond_MI[i].append(tmp_cond_MI)
            cond_MI_rank[i].append(tmp_cond_MI_rank)
            end = time.time()
            timeList.append(end-start)
        
        sumTime += max(timeList)
        timeList = []
        start = time.time()

        cond_MI_other_party_all = []
        cond_MI_other_party_list = [] # 第一个数为个数，第二个数为MI的总和
        other_party_delete_list = []

        
        for j in range(len(cond_MI[i])):
            cond_MI_other_party_list.append([0, 0]) # 第一个数为个数，第二个数为MI的总和
            other_party_delete_list.append(0) # 要删除的个数
            for k in range(len(cond_MI[i][j])):
                cond_MI_other_party_all.append((j, cond_MI[i][j][k]))

        cond_MI_other_party_all = sorted(cond_MI_other_party_all, key=lambda x: -x[1])
        
        delete_num = int((len(cond_MI_other_party_all)-new_col_num) * 0.1)
        # 遍历后delete_num个features
        for j in range(len(cond_MI_other_party_all)-delete_num, len(cond_MI_other_party_all)):
            other_party_delete_list[cond_MI_other_party_all[j][0]] += 1

        # 这里先选总和最大的，之后再调整
        for j in range(new_col_num):
            if j >= len(cond_MI_other_party_all):
                    break
            cond_MI_other_party_list[cond_MI_other_party_all[j][0]][0] += 1
            cond_MI_other_party_list[cond_MI_other_party_all[j][0]][1] += cond_MI_other_party_all[j][1]

        for j in range(len(cond_MI_other_party_list)):
            if cond_MI_other_party_list[j][0] == 0:
                cond_MI_other_party_list[j][0] = 10000
            cond_MI_other_party_list[j][1] /= cond_MI_other_party_list[j][0]

        cond_MI_other_party_arr = np.array(cond_MI_other_party_list)
        cond_MI_other_party_rank = np.argsort(-cond_MI_other_party_arr, axis=0) # 获取MI总和的排序,axis=0为按列排序，即数组间纵向比较排序,axis=1为按行排序，即数组内横向比较排序
        cond_MI_other_party_rank = list(map(lambda x:x[1], cond_MI_other_party_rank))

        next_active = cond_MI_other_party_rank[0] # 下一个发起方
        next_feature_num = cond_MI_other_party_list[next_active][0] # 下一个发起方的特征个数

        # data_party[i+1], data_party_equi[i+1] = data_party[i+1].iloc[:, cond_MI_rank[i][next_active][:next_feature_num]], data_party_equi[i+1].iloc[:, cond_MI_rank[i][next_active][:next_feature_num]]

        for j in range(i, L):
            if j == next_active+i:
                data_party[j], data_party_equi[j] = data_party[j].iloc[:, cond_MI_rank[i][j-i][:next_feature_num]], data_party_equi[j].iloc[:, cond_MI_rank[i][j-i][:next_feature_num]]
            else:
                if other_party_delete_list[j-i] != 0:
                    data_party[j], data_party_equi[j] = data_party[j].iloc[:, cond_MI_rank[i][j-i][:-other_party_delete_list[j-i]]], data_party_equi[j].iloc[:, cond_MI_rank[i][j-i][:-other_party_delete_list[j-i]]]
                else:
                    data_party[j], data_party_equi[j] = data_party[j].iloc[:, cond_MI_rank[i][j-i]], data_party_equi[j].iloc[:, cond_MI_rank[i][j-i]]


        data_party.insert(i, data_party.pop(next_active+i))
        data_party_equi.insert(i, data_party_equi.pop(next_active+i))

        if i==0:
            cross_feature[i] = getCrosstab(data_party_equi[i], target, step, model+method+str(i), RANDOMSEED, subdir)
        else:
            cross_feature[i] = pd.concat([cross_feature[i-1], getCrosstab(data_party_equi[i], target, step, model+method+str(i), RANDOMSEED, subdir)], axis=1)
        
        if i==0:
            data_sum_fl_ratio = data_party[i]
        else:
            data_sum_fl_ratio = pd.concat([data_sum_fl_ratio, data_party[i]], axis=1)
        
        new_col_num -= next_feature_num # 删除已选的个数
        
        # acc_res_fl_ratio.append(clf_cv(clf_name, data_sum_fl_ratio.copy().T.drop_duplicates().T, target))
        if new_col_num==0 or i==L-1:
            acc_res_fl_ratio_logi.append(clf_cv('logi', data_sum_fl_ratio.copy(), target_pred, RANDOMSEED))
            # acc_res_fl_ratio_svc.append(clf_cv('svc', data_sum_fl_ratio.copy(), target_pred, RANDOMSEED))
            acc_res_fl_ratio_rfc.append(clf_cv('rfc', data_sum_fl_ratio.copy(), target_pred, RANDOMSEED))
            acc_res_fl_ratio_xgbc.append(clf_cv('xgbc', data_sum_fl_ratio.copy(), target_pred, RANDOMSEED))
            acc_res_fl_ratio_dnn.append(clf_cv('dnn', data_sum_fl_ratio.copy(), target_pred, RANDOMSEED))

        end = time.time()
        sumTime += end-start

    print(sumTime)
    
    print(list(data_sum_fl_ratio.columns))
    # print(list(map(lambda x: all_columns.index(x), list(data_sum_fl_ratio.columns))))
    # print(acc_res_fl_ratio)

    print(acc_res_fl_ratio_logi)
    print(acc_res_fl_ratio_svc)
    print(acc_res_fl_ratio_rfc)
    print(acc_res_fl_ratio_xgbc)
    print(acc_res_fl_ratio_dnn)

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

# featureSelection_multirounds
def featureSelection_multirounds(new_col_num, L, data_party, data_party_equi, target, method, dataset_name, dsct_method, step, all_columns):
    data_party_selected = [[] for _ in range(L)]
    data_party_equi_selected = [[] for _ in range(L)]

    cross_feature = []
    data_sum_fl_ratio = data_party[0]
    acc_res_fl_ratio_logi = []
    acc_res_fl_ratio_svc = []
    acc_res_fl_ratio_rfc = []
    acc_res_fl_ratio_xgbc = []

    model = dataset_name+'_'+dsct_method+'_'+str(step)+'_'+str(new_col_num)

    # 并行，只考虑已有特征
    i = 0
    while True:
        if new_col_num==0:
            break
        cond_MI = [] # 用于每一轮，其他方针对active方的特征MI
        cond_MI_rank = [] # 用于每一轮，其他方针对active方的特征MI排序
        if i==0:
            cross_feature.append(getCrosstab(data_party_equi[0], target, step, model+method+str(i)))
        else:
            cross_feature.append(pd.concat([cross_feature[i-1], getCrosstab(tmp_party_equi, target, step, model+method+str(i))], axis=1))

        for j in range(1, L):
            # 计算每个参与方的每个特征的score，
            tmp_cond_MI, tmp_cond_MI_rank = calc_cond_MI(cross_feature[i], data_party_equi[j], target, method)
            cond_MI.append(tmp_cond_MI)
            cond_MI_rank.append(tmp_cond_MI_rank)

        cond_MI_other_party_all = []
        cond_MI_other_party_list = [] # 第一个数为个数，第二个数为MI的总和

        for j in range(len(cond_MI)):
            cond_MI_other_party_list.append([0, 0]) # 第一个数为个数，第二个数为MI的总和
            for k in range(len(cond_MI[j])):
                cond_MI_other_party_all.append((j, cond_MI[j][k]))

        cond_MI_other_party_all = sorted(cond_MI_other_party_all, key=lambda x: -x[1])


        # 这里先选总和最大的，之后再调整
        for j in range(new_col_num):
            cond_MI_other_party_list[cond_MI_other_party_all[j][0]][0] += 1
            cond_MI_other_party_list[cond_MI_other_party_all[j][0]][1] += cond_MI_other_party_all[j][1]

        for j in range(len(cond_MI_other_party_list)):
            if cond_MI_other_party_list[j][0] == 0:
                cond_MI_other_party_list[j][0] = 10000
            cond_MI_other_party_list[j][1] /= cond_MI_other_party_list[j][0]

        cond_MI_other_party_arr = np.array(cond_MI_other_party_list)
        cond_MI_other_party_rank = np.argsort(-cond_MI_other_party_arr, axis=0) # 获取MI总和的排序,axis=0为按列排序，即数组间纵向比较排序,axis=1为按行排序，即数组内横向比较排序
        cond_MI_other_party_rank = list(map(lambda x:x[1], cond_MI_other_party_rank))

        next_active = cond_MI_other_party_rank[0] # 下一个发起方
        next_feature_num = cond_MI_other_party_list[next_active][0] # 下一个发起方的特征个数
        if next_feature_num > new_col_num:
            next_feature_num = new_col_num

        # data_party.insert(i+1, data_party.pop(next_active+i+1))
        # data_party_equi.insert(i+1, data_party_equi.pop(next_active+i+1))

        data_party_selected[next_active+1], data_party_equi_selected[next_active+1] = data_party[next_active+1].iloc[:, cond_MI_rank[next_active][:next_feature_num]], data_party_equi[next_active+1].iloc[:, cond_MI_rank[next_active][:next_feature_num]]
        data_party[next_active+1], data_party_equi[next_active+1] = data_party[next_active+1].iloc[:, cond_MI_rank[next_active][next_feature_num:]], data_party_equi[next_active+1].iloc[:, cond_MI_rank[next_active][next_feature_num:]]
        
        tmp_party_equi = data_party_equi_selected[next_active+1].copy()

        data_sum_fl_ratio = pd.concat([data_sum_fl_ratio, data_party_selected[next_active+1]], axis=1)
        # acc_res_fl_ratio.append(clf_cv(clf_name, data_sum_fl_ratio.copy().T.drop_duplicates().T, data[target]))
        acc_res_fl_ratio_logi.append(clf_cv('logi', data_sum_fl_ratio.copy().T.drop_duplicates().T, target))
        acc_res_fl_ratio_svc.append(clf_cv('svc', data_sum_fl_ratio.copy().T.drop_duplicates().T, target))
        acc_res_fl_ratio_rfc.append(clf_cv('rfc', data_sum_fl_ratio.copy().T.drop_duplicates().T, target))
        acc_res_fl_ratio_xgbc.append(clf_cv('xgbc', data_sum_fl_ratio.copy().T.drop_duplicates().T, target))

        new_col_num -= next_feature_num # 删除已选的个数
        i += 1

    print(list(data_sum_fl_ratio.columns))
    # print(list(map(lambda x: all_columns.index(x), list(data_sum_fl_ratio.columns))))
    # print(acc_res_fl_ratio)

    print(acc_res_fl_ratio_logi)
    print(acc_res_fl_ratio_svc)
    print(acc_res_fl_ratio_rfc)
    print(acc_res_fl_ratio_xgbc)

    with open('logi_'+model+method+'.txt', 'w') as f:
        f.write(str(list(data_sum_fl_ratio.columns)))
        f.write('\r\n')
        f.write(str(list(map(lambda x: all_columns.index(x), list(data_sum_fl_ratio.columns)))))
        f.write('\r\n')
        f.write(str(acc_res_fl_ratio_logi))
        f.write('\r\n')

    with open('svc_'+model+method+'.txt', 'w') as f:
        f.write(str(list(data_sum_fl_ratio.columns)))
        f.write('\r\n')
        f.write(str(list(map(lambda x: all_columns.index(x), list(data_sum_fl_ratio.columns)))))
        f.write('\r\n')
        f.write(str(acc_res_fl_ratio_svc))
        f.write('\r\n')

    with open('rfc_'+model+method+'.txt', 'w') as f:
        f.write(str(list(data_sum_fl_ratio.columns)))
        f.write('\r\n')
        f.write(str(list(map(lambda x: all_columns.index(x), list(data_sum_fl_ratio.columns)))))
        f.write('\r\n')
        f.write(str(acc_res_fl_ratio_rfc))
        f.write('\r\n')

    with open('xgbc_'+model+method+'.txt', 'w') as f:
        f.write(str(list(data_sum_fl_ratio.columns)))
        f.write('\r\n')
        f.write(str(list(map(lambda x: all_columns.index(x), list(data_sum_fl_ratio.columns)))))
        f.write('\r\n')
        f.write(str(acc_res_fl_ratio_xgbc))
        f.write('\r\n')

# featureSelection_multirounds_del
def featureSelection_multirounds_del(new_col_num, L, data_party, data_party_equi, target, method, dataset_name, dsct_method, step, all_columns):
    data_party_selected = [[] for _ in range(L)]
    data_party_equi_selected = [[] for _ in range(L)]

    cross_feature = []
    data_sum_fl_ratio = data_party[0]
    acc_res_fl_ratio_logi = []
    acc_res_fl_ratio_svc = []
    acc_res_fl_ratio_rfc = []
    acc_res_fl_ratio_xgbc = []

    model = dataset_name+'_'+dsct_method+'_'+str(step)+'_'+str(new_col_num)

    # 并行，只考虑已有特征
    i = 0
    while True:
        if new_col_num==0:
            break
        cond_MI = [] # 用于每一轮，其他方针对active方的特征MI
        cond_MI_rank = [] # 用于每一轮，其他方针对active方的特征MI排序
        if i==0:
            cross_feature.append(getCrosstab(data_party_equi[0], target, step, model+method+str(i)))
        else:
            cross_feature.append(pd.concat([cross_feature[i-1], getCrosstab(tmp_party_equi, target, step, model+method+str(i))], axis=1))

        for j in range(1, L):
            # 计算每个参与方的每个特征的score，
            tmp_cond_MI, tmp_cond_MI_rank = calc_cond_MI(cross_feature[i], data_party_equi[j], target, method)
            cond_MI.append(tmp_cond_MI)
            cond_MI_rank.append(tmp_cond_MI_rank)

        cond_MI_other_party_all = []
        cond_MI_other_party_list = [] # 第一个数为个数，第二个数为MI的总和
        other_party_delete_list = []

        for j in range(len(cond_MI)):
            cond_MI_other_party_list.append([0, 0]) # 第一个数为个数，第二个数为MI的总和
            other_party_delete_list.append(0) # 要删除的个数
            for k in range(len(cond_MI[j])):
                cond_MI_other_party_all.append((j, cond_MI[j][k]))

        cond_MI_other_party_all = sorted(cond_MI_other_party_all, key=lambda x: -x[1])

        delete_num = int((len(cond_MI_other_party_all)-new_col_num) * 0.2)
        # 遍历后delete_num个features
        for j in range(len(cond_MI_other_party_all)-delete_num, len(cond_MI_other_party_all)):
            other_party_delete_list[cond_MI_other_party_all[j][0]] += 1

        # 这里先选总和最大的，之后再调整
        for j in range(new_col_num):
            cond_MI_other_party_list[cond_MI_other_party_all[j][0]][0] += 1
            cond_MI_other_party_list[cond_MI_other_party_all[j][0]][1] += cond_MI_other_party_all[j][1]

        for j in range(len(cond_MI_other_party_list)):
            if cond_MI_other_party_list[j][0] == 0:
                cond_MI_other_party_list[j][0] = 10000
            cond_MI_other_party_list[j][1] /= cond_MI_other_party_list[j][0]

        cond_MI_other_party_arr = np.array(cond_MI_other_party_list)
        cond_MI_other_party_rank = np.argsort(-cond_MI_other_party_arr, axis=0) # 获取MI总和的排序,axis=0为按列排序，即数组间纵向比较排序,axis=1为按行排序，即数组内横向比较排序
        cond_MI_other_party_rank = list(map(lambda x:x[1], cond_MI_other_party_rank))

        next_active = cond_MI_other_party_rank[0] # 下一个发起方
        next_feature_num = cond_MI_other_party_list[next_active][0] # 下一个发起方的特征个数
        if next_feature_num > new_col_num:
            next_feature_num = new_col_num

        # data_party.insert(i+1, data_party.pop(next_active+i+1))
        # data_party_equi.insert(i+1, data_party_equi.pop(next_active+i+1))


        data_party_selected[next_active+1], data_party_equi_selected[next_active+1] = data_party[next_active+1].iloc[:, cond_MI_rank[next_active][:next_feature_num]], data_party_equi[next_active+1].iloc[:, cond_MI_rank[next_active][:next_feature_num]]


        # 删除各方在other_party_delete_list中的features
        for j in range(1, L):
            if j == next_active+1:
                if other_party_delete_list[next_active] != 0:
                    data_party[next_active+1], data_party_equi[next_active+1] = data_party[next_active+1].iloc[:, cond_MI_rank[next_active][next_feature_num: -other_party_delete_list[j-1]]], data_party_equi[next_active+1].iloc[:, cond_MI_rank[next_active][next_feature_num: -other_party_delete_list[j-1]]]
                else:
                    data_party[next_active+1], data_party_equi[next_active+1] = data_party[next_active+1].iloc[:, cond_MI_rank[next_active][next_feature_num:]], data_party_equi[next_active+1].iloc[:, cond_MI_rank[next_active][next_feature_num:]]
            else:
                if other_party_delete_list[j-1] != 0:
                    data_party[j], data_party_equi[j] = data_party[j].iloc[:, cond_MI_rank[j-1][:-other_party_delete_list[j-1]]], data_party_equi[j].iloc[:, cond_MI_rank[j-1][:-other_party_delete_list[j-1]]]

        tmp_party_equi = data_party_equi_selected[next_active+1].copy()

        data_sum_fl_ratio = pd.concat([data_sum_fl_ratio, data_party_selected[next_active+1]], axis=1)
        # acc_res_fl_ratio.append(clf_cv(clf_name, data_sum_fl_ratio.copy().T.drop_duplicates().T, data[target]))
        acc_res_fl_ratio_logi.append(clf_cv('logi', data_sum_fl_ratio.copy(), target))
        acc_res_fl_ratio_svc.append(clf_cv('svc', data_sum_fl_ratio.copy(), target))
        acc_res_fl_ratio_rfc.append(clf_cv('rfc', data_sum_fl_ratio.copy(), target))
        acc_res_fl_ratio_xgbc.append(clf_cv('xgbc', data_sum_fl_ratio.copy(), target))

        new_col_num -= next_feature_num # 删除已选的个数
        i += 1

    print(list(data_sum_fl_ratio.columns))
    # print(list(map(lambda x: all_columns.index(x), list(data_sum_fl_ratio.columns))))
    # print(acc_res_fl_ratio)

    print(acc_res_fl_ratio_logi)
    print(acc_res_fl_ratio_svc)
    print(acc_res_fl_ratio_rfc)
    print(acc_res_fl_ratio_xgbc)

    with open('logi_'+model+method+'.txt', 'w') as f:
        f.write(str(list(data_sum_fl_ratio.columns)))
        f.write('\r\n')
        f.write(str(list(map(lambda x: all_columns.index(x), list(data_sum_fl_ratio.columns)))))
        f.write('\r\n')
        f.write(str(acc_res_fl_ratio_logi))
        f.write('\r\n')

    with open('svc_'+model+method+'.txt', 'w') as f:
        f.write(str(list(data_sum_fl_ratio.columns)))
        f.write('\r\n')
        f.write(str(list(map(lambda x: all_columns.index(x), list(data_sum_fl_ratio.columns)))))
        f.write('\r\n')
        f.write(str(acc_res_fl_ratio_svc))
        f.write('\r\n')

    with open('rfc_'+model+method+'.txt', 'w') as f:
        f.write(str(list(data_sum_fl_ratio.columns)))
        f.write('\r\n')
        f.write(str(list(map(lambda x: all_columns.index(x), list(data_sum_fl_ratio.columns)))))
        f.write('\r\n')
        f.write(str(acc_res_fl_ratio_rfc))
        f.write('\r\n')

    with open('xgbc_'+model+method+'.txt', 'w') as f:
        f.write(str(list(data_sum_fl_ratio.columns)))
        f.write('\r\n')
        f.write(str(list(map(lambda x: all_columns.index(x), list(data_sum_fl_ratio.columns)))))
        f.write('\r\n')
        f.write(str(acc_res_fl_ratio_xgbc))
        f.write('\r\n')

