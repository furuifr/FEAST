import sys
import os
sys.path.append(os.getcwd())
import pprint
import numpy as np
import pandas as pd
import time
from pprint import pprint
import random
from pandas.core.frame import DataFrame
random.seed(0)
from classification import rfc_iter, xgbc_iter, logi_iter
from discretization.dsct import equidistance_dsct, ratio_dsct, kmeans_dsct, chimerge_dsct
from preprocessing import getMimic, getScene, getMadelon, getPhysionet

def calc_MI(data_dsct, target):
    L = len(target)
    count_y = pd.crosstab(target, target)
    prob_y = count_y.sum(axis=0).div(L) # 1*2

    columns = data_dsct.columns.values.tolist()
    res = []

    for i in range(len(columns)):
        print('CMIM_pro10_single col:%s'%columns[i])
        count = pd.crosstab(data_dsct[columns[i]], target)
        prob_x = count.sum(axis=1).div(L) # L*1
        prob_xy = count.div(L) # L*2
        mutual_info = prob_xy * np.log2(prob_xy.div(np.matmul(prob_x.values.reshape(-1, 1),prob_y.values.reshape(1, -1))))
        mutual_info = mutual_info.fillna(0)
        mutual_info = mutual_info.sum(axis=1).sum()
        res.append(mutual_info)
            
    res = np.array(res)
    res_rank = np.argsort(-res)
    return list(abs(np.sort(-res))), res_rank
    

def calc_cond_MI(data_trans, data_party_dsct, target, method):

    if  method == '_FEAST_':
        return calc_cond_MI_CMIM_pro10(data_trans, data_party_dsct, target)

    elif method == '_SU_':
        return calc_cond_MI_SU(data_trans, data_party_dsct, target)

def calc_cond_MI_CMIM_pro10_ori(data_trans, data_party_dsct, target):
    L = len(target)
    min_h = 0
    count_y = pd.crosstab(target, target)
    prob_y = count_y.sum(axis=0).div(L) # 1*2
    # 其他方工作，处理data_party_dsct和中间结果
    # 对应表
    columns = data_party_dsct.columns.values.tolist()
    res = []
    res_1 = []
    data_trans_h = []
    mutual_info_y = prob_y*np.log2(prob_y)
    mutual_info_y = mutual_info_y.fillna(0)
    mutual_info_y = -mutual_info_y.sum()
    for i in range(len(columns)):
        print('CMIM_pro_10_single col:%s'%columns[i])
        col_x = []
        col_x_ori = []
        under1 = []
        under2 = []
        col_x_1 = []
        """
        count = pd.crosstab(data_party_dsct[columns[i]], target)
        prob_x = count.sum(axis=1).div(L) # L*1
        prob_xy = count.div(L) # L*2
        mutual_info = prob_xy * np.log2(prob_xy.div(np.matmul(prob_x.values.reshape(-1, 1),prob_y.values.reshape(1, -1))))
        mutual_info = mutual_info.fillna(0)
        mutual_info = mutual_info.sum(axis=1).sum()
        res.append(mutual_info)
        """
        for j in range(len(data_trans.columns)):

            # H(Y,Xj), H(Xj,Xi), H(Xj)
            count_jy = pd.crosstab(data_trans.iloc[:, j], target)
            prob_jy = count_jy.div(L)
            h_jy = prob_jy*np.log2(prob_jy)
            h_jy = h_jy.fillna(0)
            h_jy = -h_jy.sum(axis=1).sum()

            prob_j = count_jy.sum(axis=1).div(L)
            h_j = prob_j*np.log2(prob_j)
            h_j = h_j.fillna(0)
            h_j = -h_j.sum()

            count_ji = pd.crosstab(data_trans.iloc[:, j], data_party_dsct[columns[i]])
            prob_ji = count_ji.div(L)
            h_ji = prob_ji*np.log2(prob_ji)
            h_ji = h_ji.fillna(0)
            h_ji = -h_ji.sum(axis=1).sum()

            SU = h_jy+h_ji-h_j

            count_all = pd.crosstab([data_trans.iloc[:, j], data_party_dsct[columns[i]]], target)
            # count_all.to_csv('count_all_'+str(3)+'_'+str(method)+'_'+str(i)+'_'+str(j)+'.csv')
            max_row = len(count_all)*2-(count_all==0).sum().sum()

            # 计算x_i,x_j的联合概率
            prob_xij = count_all.sum(axis=1).div(L) # L*1
            prob_xij = pd.concat([prob_xij, prob_xij],axis=1)
            # 计算x_i,x_j,y的联合概率
            prob_mutual = count_all.div(L) # L*2
            max_row = len(prob_mutual)*2-(prob_mutual == 0).sum().sum()

            count_jy = pd.crosstab(data_trans.iloc[:, j], target)
            list_jy = list(count_all.index.labels[0])
            count_jy_0 = list(map(lambda x: count_jy.iat[x,0], list_jy))
            count_jy_1 = list(map(lambda x: count_jy.iat[x,1], list_jy))
            count_jy_pro = pd.DataFrame({'0':count_jy_0, '1':count_jy_1})

            prob_jy = count_jy_pro.div(L) # L*2
            prob_j = count_jy_pro.sum(axis=1).div(L) # L*1
            prob_j = pd.concat([prob_j, prob_j],axis=1)
    
            mutual_info = prob_mutual*np.log2((prob_mutual.values*prob_j.values)/(prob_xij.values*prob_jy.values))
            mutual_info = mutual_info.fillna(0)
            max_row1 = len(mutual_info)
            # max_row = len(mutual_info)
            mutual_info = mutual_info.sum(axis=1).sum()
            col_x_ori.append(mutual_info)
            under1.append(SU*max_row**0.5)
            under2.append(-((prob_mutual)*np.log2(prob_mutual)).sum(axis=1).sum()*(max_row1**0.5))
            col_x_1.append(mutual_info/(-((prob_mutual)*np.log2(prob_mutual)).sum(axis=1).sum()*(max_row1**0.5)))
            if SU*max_row**0.5==0:
                col_x.append(0)
            else:
                col_x.append(mutual_info/(SU*max_row**0.5))
            
            # prob_jy = count_jy.div(L) # L*2
            # prob_j = count_jy.sum(axis=1).div(L) # L*1
            # prob_j = pd.concat([prob_j, prob_j],axis=1)

            # mutual_info_ori = -(prob_jy*np.log2(prob_jy.values/prob_j.values))
            # mutual_info_ori = mutual_info_ori.fillna(0)
            # mutual_info_ori = mutual_info_ori.sum(axis=1).sum()

            # mutual_info_add = -(prob_mutual*np.log2(prob_mutual.values/prob_xij.values))
            # mutual_info_add = mutual_info_add.fillna(0)
            # mutual_info_add = mutual_info_add.sum(axis=1).sum()    

            # mutual_info_1 = mutual_info_ori - mutual_info_add
            pass

        res.append(sum(col_x))
        res_1.append(sum(col_x_1))
    
    res = np.array(res)
    res_rank = np.argsort(-res)

    res_1 = np.array(res_1)
    res_rank_1 = np.argsort(-res_1)

    return list(abs(np.sort(-res))), res_rank

# calc_cond_MI_CMIM_pro10_dyn 删除小于等于中位数对应值的一半的所有feature
def calc_cond_MI_CMIM_pro10(data_trans, data_party_equi, target):
    L = len(target)
    count_y = pd.crosstab(target, target)
    prob_y = count_y.sum(axis=0).div(L) # 1*2
    # 其他方工作，处理data_party_equi和中间结果
    # 对应表
    columns = data_party_equi.columns.values.tolist()
    res = []
    res_min = []
    mutual_info_y = prob_y*np.log2(prob_y)
    mutual_info_y = mutual_info_y.fillna(0)
    mutual_info_y = -mutual_info_y.sum()
    for i in range(len(columns)):
        print('CMIM_pro_10_FL_multi(single) col:%s'%columns[i])
        col_x = []
        col_x_ori = []

        """
        count = pd.crosstab(data_party_equi[columns[i]], target)
        prob_x = count.sum(axis=1).div(L) # L*1
        prob_xy = count.div(L) # L*2
        mutual_info = prob_xy * np.log2(prob_xy.div(np.matmul(prob_x.values.reshape(-1, 1),prob_y.values.reshape(1, -1))))
        mutual_info = mutual_info.fillna(0)
        mutual_info = mutual_info.sum(axis=1).sum()
        res.append(mutual_info)
        """
        for j in range(len(data_trans.columns)):

            # H(Y,Xj), H(Xj,Xi), H(Xj)
            count_jy = pd.crosstab(data_trans.iloc[:, j], target)
            prob_jy = count_jy.div(L)
            h_jy = prob_jy*np.log2(prob_jy)
            h_jy = h_jy.fillna(0)
            h_jy = -h_jy.sum(axis=1).sum()

            prob_j = count_jy.sum(axis=1).div(L)
            h_j = prob_j*np.log2(prob_j)
            h_j = h_j.fillna(0)
            h_j = -h_j.sum()

            count_ji = pd.crosstab(data_trans.iloc[:, j], data_party_equi[columns[i]])
            prob_ji = count_ji.div(L)
            h_ji = prob_ji*np.log2(prob_ji)
            h_ji = h_ji.fillna(0)
            h_ji = -h_ji.sum(axis=1).sum()

            SU = h_jy+h_ji-h_j

            count_all = pd.crosstab([data_trans.iloc[:, j], data_party_equi[columns[i]]], target)

            # 计算x_i,x_j的联合概率
            prob_xij = count_all.sum(axis=1).div(L) # L*1
            prob_xij = pd.concat([prob_xij, prob_xij],axis=1)
            # 计算x_i,x_j,y的联合概率
            prob_mutual = count_all.div(L) # L*2
            max_row = len(prob_mutual)*2-(prob_mutual == 0).sum().sum()

            count_jy = pd.crosstab(data_trans.iloc[:, j], target)
            list_jy = list(count_all.index.labels[0])
            count_jy_0 = list(map(lambda x: count_jy.iat[x,0], list_jy))
            count_jy_1 = list(map(lambda x: count_jy.iat[x,1], list_jy))
            count_jy_pro = pd.DataFrame({'0':count_jy_0, '1':count_jy_1})

            prob_jy = count_jy_pro.div(L) # L*2
            prob_j = count_jy_pro.sum(axis=1).div(L) # L*1
            prob_j = pd.concat([prob_j, prob_j],axis=1)
    
            mutual_info = prob_mutual*np.log2((prob_mutual.values*prob_j.values)/(prob_xij.values*prob_jy.values))
            mutual_info = mutual_info.fillna(0)
            mutual_info = mutual_info.sum(axis=1).sum()
            col_x_ori.append(mutual_info)
            # if SU*max_row**0.5==0:
            #     col_x.append(0)
            # else:
            #     col_x.append(mutual_info/(SU*max_row**0.5))

            col_x.append(mutual_info)

            pass
        
        res.append(sum(col_x))
        res_min.append(min(col_x))

    res = np.array(res)
    res_rank = np.argsort(-res)

    res_min = np.array(res_min)
    res_min_rank = np.argsort(-res_min)
    threshold = res_min[res_min_rank[int(len(res_min_rank)/2)]]/2
    # del_num = int(len(res_rank)*0.2)
    del_list = []
    for i in range(len(res_rank)):
        if res_min[res_min_rank[-(i+1)]] <= threshold:
            del_list.append(res_min_rank[-(i+1)])
        else:
            break
    del_list.sort(key=lambda x: -x)
    res = list(res)
    res_rank = list(res_rank)
    for i in range(len(del_list)):
        res.pop(del_list[i])
        res_rank.pop(list(res_rank).index(del_list[i]))
    
    res = np.array(res)
    res_rank = np.array(res_rank)

    return list(abs(np.sort(-res))), res_rank

def calc_cond_MI_SU(data_trans, data_party_dsct, target):
    L = len(target)
    min_h = 0
    count_y = pd.crosstab(target, target)
    prob_y = count_y.sum(axis=0).div(L) # 1*2
    # 其他方工作，处理data_party_dsct和中间结果
    # 对应表
    columns = data_party_dsct.columns.values.tolist()
    res = []
    data_trans_h = []
    mutual_info_y = prob_y*np.log2(prob_y)
    mutual_info_y = mutual_info_y.fillna(0)
    mutual_info_y = -mutual_info_y.sum()
    for i in range(len(columns)):
        print('SU_single col:%s'%columns[i])
        col_x = []
        # col_x_1 = []
        """
        count = pd.crosstab(data_party_dsct[columns[i]], target)
        prob_x = count.sum(axis=1).div(L) # L*1
        prob_xy = count.div(L) # L*2
        mutual_info = prob_xy * np.log2(prob_xy.div(np.matmul(prob_x.values.reshape(-1, 1),prob_y.values.reshape(1, -1))))
        mutual_info = mutual_info.fillna(0)
        mutual_info = mutual_info.sum(axis=1).sum()
        res.append(mutual_info)
        """
        for j in range(len(data_trans.columns)):

            # H(Y,Xj), H(Xj,Xi), H(Xj)
            count_jy = pd.crosstab(data_trans.iloc[:, j], target)
            prob_jy = count_jy.div(L)
            h_jy = prob_jy*np.log2(prob_jy)
            h_jy = h_jy.fillna(0)
            h_jy = -h_jy.sum(axis=1).sum()

            prob_j = count_jy.sum(axis=1).div(L)
            h_j = prob_j*np.log2(prob_j)
            h_j = h_j.fillna(0)
            h_j = -h_j.sum()

            count_ji = pd.crosstab(data_trans.iloc[:, j], data_party_dsct[columns[i]])
            prob_ji = count_ji.div(L)
            h_ji = prob_ji*np.log2(prob_ji)
            h_ji = h_ji.fillna(0)
            h_ji = -h_ji.sum(axis=1).sum()


            SU = h_jy+h_ji-h_j

            count_all = pd.crosstab([data_trans.iloc[:, j], data_party_dsct[columns[i]]], target)
            # count_all.to_csv('count_all_'+str(3)+'_'+str(method)+'_'+str(i)+'_'+str(j)+'.csv')

            # 计算x_i,x_j的联合概率
            prob_xij = count_all.sum(axis=1).div(L) # L*1
            prob_xij = pd.concat([prob_xij, prob_xij],axis=1)
            # 计算x_i,x_j,y的联合概率
            prob_mutual = count_all.div(L) # L*2

            count_jy = pd.crosstab(data_trans.iloc[:, j], target)
            list_jy = list(count_all.index.labels[0])
            count_jy_0 = list(map(lambda x: count_jy.iat[x,0], list_jy))
            count_jy_1 = list(map(lambda x: count_jy.iat[x,1], list_jy))
            count_jy_pro = pd.DataFrame({'0':count_jy_0, '1':count_jy_1})

            prob_jy = count_jy_pro.div(L) # L*2
            prob_j = count_jy_pro.sum(axis=1).div(L) # L*1
            prob_j = pd.concat([prob_j, prob_j],axis=1)
    
            mutual_info = prob_mutual*np.log2((prob_mutual.values*prob_j.values)/(prob_xij.values*prob_jy.values))
            max_rol = len(mutual_info)
            mutual_info = mutual_info.fillna(0)
            mutual_info = mutual_info.sum(axis=1).sum()

            col_x.append(mutual_info/SU)

        res.append(sum(col_x))
    
    res = np.array(res)
    res_rank = np.argsort(-res)

    return list(abs(np.sort(-res))), res_rank
