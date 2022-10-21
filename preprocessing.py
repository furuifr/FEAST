from tokenize import String
from matplotlib.pyplot import flag
import numpy as np
from numpy.core.numeric import NaN
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import re


def read_data(path='../dataset/avazu-ctr-prediction/train.csv'):
    df = pd.read_csv(path, chunksize=1000000)
    # for data in df:
    #     print(data.head())
    return df

def getData(dataset_name, task=1):
    if dataset_name=="physionet":
        return getPhysionet(task)
    elif dataset_name=="mimic":
        return getMimic()
    elif dataset_name=="census":
        return getCensus()
    elif dataset_name=="nomao":
        return getNomao()

def getBoston():
    boston = load_boston()
    data = boston.data
    target = boston.target
    data = pd.DataFrame(data, columns = boston.feature_names)
    target = pd.DataFrame(target, columns = ['TARGET'])
    data = pd.concat([data, target], axis=1)
    return data
    
def getMimic():
    # data = pd.read_csv('./dataset/mimic/MICU_admits_clean_guest.csv')
    data = pd.read_csv('./dataset/mimic/MICU_admits_clean_modify.csv')
    data = (data-data.min())/(data.max()-data.min())
    data = data.fillna(0)
    return data

def getCensus():
    data = pd.read_csv('./dataset/census/census-income-clean.csv')
    data = (data-data.min())/(data.max()-data.min())
    return data

def getPhysionet1():
    data = pd.read_csv('./dataset/Physionet2012/physionet1_p1.csv')
    return data

def getPhysionet(task):
    task = task+46
    train = pd.read_csv('./dataset/Physionet2012/train_set.csv')
    valid = pd.read_csv('./dataset/Physionet2012/valid_set.csv')
    test = pd.read_csv('./dataset/Physionet2012/test_set.csv')
    train = pd.concat([train, valid])
    train = pd.concat([train, test])
    train = train.reset_index(drop=True)
    data = train.iloc[:, 2:43]
    target = train.iloc[:, task]
    data = pd.concat([data, target], axis=1)
    return data

def getNomao():
    data = pd.read_csv('./dataset/nomao/Nomao-clean.csv')
    return data

def getScene():
    data = pd.read_csv('./dataset/openML/scene.csv')
    return data

def getMadelon():
    data = pd.read_csv('./dataset/openML/madelon.csv')
    return data

def getYearPrediction():
    data = pd.read_csv('./dataset/YearPrediction/YearPredictionMSD.csv')
    return data   


def get_column(df):
    columns = []
    for data in df:
        for col in data.columns:
            columns.append(col)
        break
    return columns

# 离散化日期数据，日期数据是有规律可言的
def discretize_date(column):
    return


# 根据熵值差计算连续值离散化的分割点
def discretize_entropy(column):
    
    return

def data_clean(path):
    df = pd.read_csv(path)
    # 首先删除空缺值较多的列以及无关列
    # ...
    # 将类别数据转换为数字类别
    columns = df.columns.values.tolist()
    for col in columns:
        values_type = df[col].map(lambda x: type(x)).unique()
        if str in values_type:
            values = df[col].unique().tolist()
            values.sort()
            value_mapping = {}
            index = 0
            for value in values:
                value_mapping[value] = index
                index += 1
            df[col] = df[col].map(value_mapping)
    df.to_csv('./dataset/nomao/Nomao-clean.csv', index=None)
        
def convert_str_to_num(df, col, values):
    pat1 = "[-+]?[0-9]*\.[0-9]+"
    pat2 = "\d+"
    sum = 0
    count = 0
    for i in range(len(df[col])):
        print(i)
        # df[col][i]为数字或者字符串形的数字
        if type(df[col][i]) == float or re.fullmatch(pat1, df[col][i]) is not None or re.fullmatch(pat2, df[col][i]) is not None:
            sum += float(df[col][i])
            count += 1
    aver = sum/count
    value_mapping = {}
    for value in values:
        # value为字符串，且不为数字
        if type(value) == str and re.fullmatch(pat1, value) is None and re.fullmatch(pat2, value) is None:
            value_mapping[value] = aver
        else:
            value_mapping[value] = float(value)
    df[col] = df[col].map(value_mapping)

def nomao_data_clean(path):
    df = pd.read_csv(path)
    # 首先删除空缺值较多的列以及无关列
    # ...
    # 将类别数据转换为数字类别
    pat1 = "[-+]?[0-9]*\.[0-9]+"
    pat2 = "\d+"
    columns = df.columns.values.tolist()
    for col in columns:
        values_type = df[col].map(lambda x: type(x)).unique()
        if str in values_type:
            values = df[col].unique().tolist()
            flag = False
            for value in values:
                # 有字符串形的数字
                if type(value) == str and (re.fullmatch(pat1, value) is not None or re.fullmatch(pat2, value) is not None):
                    convert_str_to_num(df, col, values)
                    flag = True
                    break
            if flag:
                continue
            values.sort()
            value_mapping = {}
            index = 0
            for value in values:
                value_mapping[value] = index
                index += 1
            df[col] = df[col].map(value_mapping)
    df.to_csv('./dataset/nomao/Nomao-clean.csv', index=None)
        

# 依据crosstab进行分层抽样，从而减少行的数量
def sampling(data, target, sampling_method, bin, step, RANDOMSEED):
    # row, col, bin分箱数, step合并数
    # objective: min{row*(col/step)}, col/step取上界
    # 限制条件：bin^step * 2 < row

    row = len(data)
    col = len(data.columns)

    new_row = bin**(step+1)*2*1 # 1之后通过实验再调整
    if new_row>row:
        new_row = row

    # new_row = 6000

    if sampling_method == "stratified" and new_row != row:
        X_train, X_test, y_train, y_test = train_test_split(data, target, train_size=new_row, stratify=target, random_state=RANDOMSEED)
        return X_train.reset_index(drop=True), y_train.reset_index(drop=True)

    elif sampling_method == "random":
        X_train, X_test, y_train, y_test = train_test_split(data, target, train_size=new_row, stratify=None, random_state=RANDOMSEED)
        return X_train.reset_index(drop=True), y_train.reset_index(drop=True)
    
    else:
        return data, target

# 依据crosstab进行分层抽样，从而减少行的数量，获取采样后的行数
def getSamplingRow(data: pd.DataFrame, target: pd.DataFrame, sampling_method: String, bin: int, step: int, RANDOMSEED: int) -> list:
    # row, col, bin分箱数, step合并数
    # objective: min{row*(col/step)}, col/step取上界
    # 限制条件：bin^step * 2 < row

    row = len(data)
    col = len(data.columns)

    new_row = bin**(step+1)*2*1 # 1之后通过实验再调整
    if new_row>row:
        new_row = row

    if sampling_method == "stratified":
        X_train, X_test, y_train, y_test = train_test_split(data, target, train_size=new_row, stratify=target, random_state=RANDOMSEED)
        return list(X_train.axes[0])

    elif sampling_method == "random":
        X_train, X_test, y_train, y_test = train_test_split(data, target, train_size=new_row, stratify=None, random_state=RANDOMSEED)
        return list(X_train.axes[0])
    
    else:
        return list(data.iloc[:, 0])


if __name__ == '__main__':
    # nomao_data_clean('./dataset/nomao/Nomao.csv')
    data = pd.read_csv('./dataset/nomao/Nomao-clean.csv')
