import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score
from preprocessing import getBoston, getScene, getMadelon, getMimic
from pprint import pprint
from random import choice, random
import sys  # 导入sys模块
sys.setrecursionlimit(3000)  # 将默认的递归深度修改为3000

rfc_parameter = {
    "n_estimators":[500,800,1000],
    "criterion":['entropy'],
    "max_features":['auto','sqrt','log2',None],
    "min_samples_leaf":[2,3,4],
    "min_samples_split":[2,3,4],
    "oob_score":[True, False],
    "max_depth":[None,10,20,50,100],
}

def parameter_search():
    parameters = []
    variable = rfc_parameter # rfc参数选项
    for param in variable.keys():
        parameters.append(choice(variable[param]))
    return parameters

def rfc_choice_param(data, target):
    rfc_list = []
    for i in range(20): 
        n, criterion, max_features, min_samples_leaf, min_samples_split, oob_score, max_depth = parameter_search()
        rfc = RandomForestClassifier(n_estimators=n, criterion=criterion, max_features=max_features, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, oob_score=oob_score, max_depth=max_depth, n_jobs=5)
        score = cross_val_score(rfc, data, target, cv=10, scoring='roc_auc', n_jobs=5).mean()
        print(i, score, n, criterion, max_features, min_samples_leaf, min_samples_split, oob_score, max_depth)
        rfc_list.append((score, n, criterion, max_features, min_samples_leaf, min_samples_split, oob_score, max_depth))
    rfc_list.sort(key=lambda x:-x[0])
    return rfc_list

def logi_iter(data, target):
    # print(data.info())
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, target, test_size=0.3)
    logi = LogisticRegression(n_jobs=5)
    logi = logi.fit(Xtrain, Ytrain)
    score = roc_auc_score(Ytest, logi.predict_proba(Xtest)[:,1])
    importance = abs(logi.coef_[0])
    # print(importance)
    rank = np.argsort(-importance)
    return score, rank

def logi_cv(data, target, RANDOMSEED):
    logi = LogisticRegression(n_jobs=5, random_state=RANDOMSEED)
    score = cross_val_score(logi, data, target, cv=5, scoring='roc_auc', n_jobs=5).mean()
    return score

def xgbc_iter(data, target):
    # print(data.info())
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, target, test_size=0.3)
    xgbc = XGBClassifier(n_estimators=500, n_jobs=5)
    xgbc = xgbc.fit(Xtrain, Ytrain)
    score = roc_auc_score(Ytest, xgbc.predict_proba(Xtest)[:,1])
    importance = xgbc.feature_importances_
    # print(importance)
    rank = np.argsort(-importance)
    return score, rank

def xgbc_cv(data, target, RANDOMSEED):
    xgbc = XGBClassifier(n_estimators=200, n_jobs=5, random_state=RANDOMSEED, learning_rate=0.1)
    score = cross_val_score(xgbc, data, target, cv=5, scoring='roc_auc', n_jobs=5).mean()
    return score


def rfc_cv(data, target, RANDOMSEED):
    rfc = RandomForestClassifier(n_estimators=200, criterion='entropy', max_features='auto', min_samples_leaf=4, min_samples_split=2, oob_score=True, max_depth=20, n_jobs=5, random_state=RANDOMSEED)
    score = cross_val_score(rfc, data, target, cv=5, scoring='roc_auc', n_jobs=5).mean()
    return score

def rfc_iter(data, target):
    # print(data.info())
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, target, test_size=0.3)
    rfc = RandomForestClassifier(n_estimators=500, criterion='entropy', max_features='auto', min_samples_leaf=4, min_samples_split=2, oob_score=True, max_depth=20, n_jobs=5)
    rfc = rfc.fit(Xtrain, Ytrain)
    score = roc_auc_score(Ytest, rfc.predict_proba(Xtest)[:,1])
    importance = rfc.feature_importances_
    print(importance)
    rank = np.argsort(-importance)
    return score, rank

def svc_cv(data, target, RANDOMSEED):
    svc = SVC(C=0.5, kernel='rbf', gamma='auto', random_state=RANDOMSEED)
    score = cross_val_score(svc, data, target, cv=5, scoring='roc_auc', n_jobs=5).mean()
    return score

def clf_cv(clf_name, data, target, RANDOMSEED):
    if clf_name=='rfc':
        return rfc_cv(data, target, RANDOMSEED)
    elif clf_name=='xgbc':
        return xgbc_cv(data, target, RANDOMSEED)
    elif clf_name=='logi':
        return logi_cv(data, target, RANDOMSEED)
    elif clf_name=='svc':
        return svc_cv(data, target, RANDOMSEED)
    

if __name__ == '__main__':
    # data = getBoston()
    data = getMimic()
    # data = getMadelon()
    pprint(rfc_choice_param(data.iloc[:, :-1],data.iloc[:,-1]))
    # data = equidistance_dsct(data)
    # data = entropy_dsct(data)
    # rfc_cv(data, 'Urban')

