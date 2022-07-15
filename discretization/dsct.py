import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans   #导入kmeans
import sys
import os
# from discretization.Entropy_dsct import Entropy_Discretization
from preprocessing import getMimic, getScene, getMadelon, getPhysionet
import time

def dsct(dsct_method, data, k=10, target_name='', target=None):
    if dsct_method=='equidistance':
        return equidistance_dsct(data, k)
    elif dsct_method=='ratio':
        return ratio_dsct(data, k)
    elif dsct_method=='kmeans':
        return kmeans_dsct(data, k)
    elif dsct_method=='mdlp':
        return mdlp_dsct2(data, target)
    elif dsct_method=='chimerge':
        return chimerge_dsct(data, target_name, target)
    elif dsct_method=='bestks':
        return bestks_dsct(data, target_name, target)


# 试一下本来就是分类变量的会不会再次分类，如试试‘Urban’会不会被分类
def equidistance_dsct(data , q=10):
    L = len(data)
    q = max(min(L//3, q), 2)
    cols = data.columns.values.tolist()
    for col in cols:
        print('equidistance_dsct col:%s'%col)
        if len(set(data[col]))<q:
            qq = len(set(data[col]))
        else:
            qq = q
        data.loc[:, col+'_dsct'] = pd.cut(data[col], qq, labels=False) # labels=False只显示数据位于第几个箱子里
        data = data.drop(columns=col)
    return data

def ratio_dsct(data, k=10):
    L = len(data)
    cols = data.columns.values.tolist()
    for col in cols:
        print('ratio_dsct col:%s'%col)
        if len(set(data[col]))<=k:
            kk = len(set(data[col]))
            data.loc[:, col+'_dsct'] = pd.cut(data[col], kk, labels=False) # labels=False只显示数据位于第几个箱子里
        else:
            kk = k
            w = [i/kk for i in range(kk+1)] # 计算百分比
            # w = sorted(list(set(map(lambda x:round(data[col].quantile(x),9), w))))
            w = sorted(list(set(map(lambda x:data[col].quantile(x), w))))
            # w = sorted(list(set(data[col].describe(percentiles=w)[4:-1]))) # 使用describe分组，会默认有50%这个选项，所以分为3组的时候其实会默认4组
            w[0] -= 0.01
            data.loc[:, col+'_dsct'] = pd.cut(data[col], w, labels=False) # labels=False只显示数据位于第几个箱子里

        data = data.drop(columns=col)
    return data

def kmeans_dsct(data, k=10):
    L = len(data)
    cols = data.columns.values.tolist()
    for col in cols:
        print('kmeans_dsct col:%s'%col)
        if len(set(data[col]))<k:
            kk = len(set(data[col]))
            data.loc[:, col+'_dsct'] = pd.cut(data[col], kk, labels=False) # labels=False只显示数据位于第几个箱子里
        else:
            kmodel = KMeans(n_clusters = k)      #确定族数
            kmodel.fit(data[col].values.reshape(len(data[col]),1))    #训练数据集
            c = pd.DataFrame(np.sort(kmodel.cluster_centers_))   #确定中心并排序
            w = c.rolling(2).mean().iloc[1:]      #取移动平均值
            w = [-0.01]+list(w[0])+[data[col].max()+0.01]       #加上最大最小值作为边界值
            w = list(np.sort(w))          #再次排序
            data.loc[:, col+'_dsct'] = pd.cut(data[col], w, labels = False)

        data = data.drop(columns=col)
    return data   

def mdlp_dsct1(data, target):
    from mdlp.discretization import MDLP
    L = len(data)
    cols = data.columns.values.tolist()
    cols = list(map(lambda x:x+'_dsct', cols))
    mdlp = MDLP()
    data_dsct = mdlp.fit_transform(np.array(data), np.array(target))
    return pd.DataFrame(data_dsct, columns=cols)
    
def mdlp_dsct2(data_copy, target):
    from discretization.entropy_mdlp import MDLP
    L = len(data_copy)
    cols = data_copy.columns.values.tolist()
    for col in cols:
        print('mdlp_dsct col:%s'%col)
        mdlp = MDLP()
        w = mdlp.cut_points(np.array(data_copy[col]), np.array(target))
        if w.size:
            w=list(w)
            w.insert(0, min(data_copy[col])-0.01)
            w.insert(len(w), max(data_copy[col])+0.01)
            data_copy.loc[:, col+'_dsct'] = pd.cut(data_copy[col], w, labels = False)
        else:
            data_copy.loc[:, col+'_dsct'] = pd.Series([0]*len(data_copy[col]))
        data_copy = data_copy.drop(columns=col)
    return data_copy

def chimerge_dsct(data_copy, target_name, target):
    from discretization.chimerge import ChiMerge
    L = len(data_copy)
    cols = data_copy.columns.values.tolist()
    data_copy[target_name] = target
    for col in cols:
        print('chimerge_dsct col:%s'%col)
        if col[-2:] == '_1':
            data_copy.loc[:, col+'_dsct'] = data_copy[col]
            data_copy = data_copy.drop(columns=col)
            continue
        w = ChiMerge(data_copy, col, target_name, confidenceVal=3.841, bin=10, sample=None).iloc[:, 1]
        if w.size>1:
            w = list(w)
            if min(data_copy[col]) == w[0]:
                w[0] -= 0.01
            else:
                w.insert(0, min(data_copy[col])-0.01)
            if max(data_copy[col]) == w[len(w)-1]:
                w[len(w)-1] += 0.01
            else:
                w.insert(len(w), max(data_copy[col])+0.01)
            data_copy.loc[:, col+'_dsct'] = pd.cut(data_copy[col], w, labels = False)
        else:
            data_copy.loc[:, col+'_dsct'] = pd.Series([0]*len(data_copy[col]))
        data_copy = data_copy.drop(columns=col)
    data_copy = data_copy.drop([target_name], axis=1)
    return data_copy

def bestks_dsct(data_copy, target_name, target):
    from discretization.bestks import cut_main_fun
    from discretization.bestks import univeral_df
    L = len(data_copy)
    cols = data_copy.columns.values.tolist()
    data_copy[target_name] = target
    for col in cols:
        print('bestks_dsct col:%s'%col)
        if col == target_name:
            data_copy = data_copy.drop(columns=col)
            break
        result=univeral_df(data_copy, col, target_name, 'total', 'good', 'bad')
        w = cut_main_fun(data_df=result, feature=col, rate=0.05, total_name='total', good_name='good', bad_name='bad')
        data_copy.loc[:, col+'_dsct'] = pd.cut(data_copy[col], w, labels = False)
        data_copy = data_copy.drop(columns=col)
    data_copy = data_copy.drop([target_name], axis=1)
    return data_copy


if __name__ == '__main__':
    data = getMimic()
    target = 'future_readmit'
    # time_start=time.time()
    # data_equi1 = mdlp_dsct1(data.copy(), target)
    # print('data_equi1 cost: %f'%(time.time()-time_start)) 
    # time_start=time.time()
    # data_equi2 = mdlp_dsct2(data.copy(), data[target])
    # print('data_equi2 cost: %f'%(time.time()-time_start)) 
    # for i in range(len(data.columns.values.tolist())):
    #     print(set(data_equi1.iloc[:,i]))
    #     print(set(data_equi2.iloc[:,i]))
    data_equi = bestks_dsct(data.copy(), target)

    pass
