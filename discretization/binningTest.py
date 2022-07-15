import warnings
warnings.filterwarnings("ignore")
import sys
import os
sys.path.append(os.getcwd())
from preprocessing import getMimic, getScene, getMadelon, getPhysionet
from discretization.dsct import equidistance_dsct, mdlp_dsct2, ratio_dsct, kmeans_dsct, chimerge_dsct, bestks_dsct
from classification import logi_cv, rfc_cv, xgbc_cv
from pprint import pprint
import numpy as np
import pandas as pd

if __name__ == '__main__':
    data = getMimic()
    target = 'future_readmit'
    
    result = []

    # # 原数据
    # print("----------原数据----------")
    # tmp = []
    # tmp.append(logi_cv(data.iloc[:, :-1], data[target]))
    # tmp.append(rfc_cv(data.iloc[:, :-1], data[target]))
    # tmp.append(xgbc_cv(data.iloc[:, :-1], data[target]))
    # result.append(tmp)

    # print("----------等距分箱----------")
    # data_equi = equidistance_dsct(data.copy())
    # tmp = []
    # tmp.append(logi_cv(data_equi.iloc[:, :-1], data[target]))
    # tmp.append(rfc_cv(data_equi.iloc[:, :-1], data[target]))
    # tmp.append(xgbc_cv(data_equi.iloc[:, :-1], data[target]))
    # result.append(tmp)

    # print("----------等频分箱----------")
    # data_equi = ratio_dsct(data.copy())
    # tmp = []
    # tmp.append(logi_cv(data_equi.iloc[:, :-1], data[target]))
    # tmp.append(rfc_cv(data_equi.iloc[:, :-1], data[target]))
    # tmp.append(xgbc_cv(data_equi.iloc[:, :-1], data[target]))
    # result.append(tmp)

    # print("----------聚类分箱----------")
    # data_equi = kmeans_dsct(data.copy())
    # tmp = []
    # tmp.append(logi_cv(data_equi.iloc[:, :-1], data[target]))
    # tmp.append(rfc_cv(data_equi.iloc[:, :-1], data[target]))
    # tmp.append(xgbc_cv(data_equi.iloc[:, :-1], data[target]))
    # result.append(tmp)
    
    # print("----------信息熵分箱----------")
    # data_equi = mdlp_dsct2(data.copy(), target)
    # tmp = []
    # tmp.append(logi_cv(data_equi.iloc[:, :-1], data[target]))
    # tmp.append(rfc_cv(data_equi.iloc[:, :-1], data[target]))
    # tmp.append(xgbc_cv(data_equi.iloc[:, :-1], data[target]))
    # result.append(tmp)

    # print("----------卡方分箱----------")
    # data_equi = chimerge_dsct(data.copy(), target)
    # tmp = []
    # tmp.append(logi_cv(data_equi.iloc[:, :-1], data[target]))
    # tmp.append(rfc_cv(data_equi.iloc[:, :-1], data[target]))
    # tmp.append(xgbc_cv(data_equi.iloc[:, :-1], data[target]))
    # result.append(tmp)
    
    # print("----------best-ks分箱----------")
    # data_equi = bestks_dsct(data.copy(), target)
    # tmp = []
    # tmp.append(logi_cv(data_equi.iloc[:, :-1], data[target]))
    # tmp.append(rfc_cv(data_equi.iloc[:, :-1], data[target]))
    # tmp.append(xgbc_cv(data_equi.iloc[:, :-1], data[target]))
    # result.append(tmp)
    
    # np.save('dsct_result.npy', np.array(result))

    # pprint(result)

    print("----------信息熵分箱----------")
    data_equi = mdlp_dsct2(data.copy(), target)
    cols = data_equi.columns.values.tolist()
    for col in cols:
        print('mdlp_dsct col:%s, %d'%(col, len(set(data_equi[col]))))

    print("----------卡方分箱----------")
    data_equi = chimerge_dsct(data.copy(), target)
    cols = data_equi.columns.values.tolist()
    for col in cols:
        print('chimerge_dsct col:%s, %d'%(col, len(set(data_equi[col]))))
    
    print("----------best-ks分箱----------")
    data_equi = bestks_dsct(data.copy(), target)
    cols = data_equi.columns.values.tolist()
    for col in cols:
        print('bestks_dsct col:%s, %d'%(col, len(set(data_equi[col]))))