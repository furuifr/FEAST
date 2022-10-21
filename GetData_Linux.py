from fileinput import filename
import imp
import os
import numpy as np
import pandas as pd
import re

def subdir(filepath):
    pathDir =  os.listdir(filepath)
    # pathDir.sort(key=lambda x:int(x[:-4]))
    for allDir in pathDir:
        child = os.path.join(filepath, allDir)
        if os.path.isfile(child):
            if os.path.splitext(child)[-1] != '.txt':
                continue
            readFile(child)
            print(child)
            continue
        subdir(child)
    
def readFile(filenames):
    name_list = filenames.split("_")
    type = name_list[0].split("/")[-1]
    with open(filenames, 'r') as f:
        lines = f.readlines()
        lnum = 0
        for line in lines:
            lnum += 1
            if lnum == 3:
                print(line)
                auc_list = list(line[1:-2].split(","))
                auc = round(float(auc_list[-1]), 4)
                print(auc)
                if type == 'logi':
                    logi_res.loc[int(name_list[-3])][name_list[-2]] = auc
                elif type == 'rfc':
                    rfc_res.loc[int(name_list[-3])][name_list[-2]] = auc
                elif type == 'svc':
                    svc_res.loc[int(name_list[-3])][name_list[-2]] = auc
                elif type == 'xgbc':
                    xgbc_res.loc[int(name_list[-3])][name_list[-2]] = auc
                elif type == 'dnn':
                    dnn_res.loc[int(name_list[-3])][name_list[-2]] = auc
        
if __name__ == "__main__":
    rootdir = '/home/furui/code/SIGMOD/FeatureSelection/mimic0/del0.1/1'
    # columns = {'FEAST', 'CFEAST', 'IV', 'Lasso', 'MI', 'Random'}
    columns = {'FEAST'}
    index = [(i+1) for i in range(136)]
    dnn_res = pd.DataFrame(columns=columns, index=index)
    logi_res = pd.DataFrame(columns=columns, index=index)
    rfc_res = pd.DataFrame(columns=columns, index=index)
    xgbc_res = pd.DataFrame(columns=columns, index=index)
    svc_res = pd.DataFrame(columns=columns, index=index)
    subdir(rootdir)
    dnn_res.to_csv(rootdir+'/dnn_res.csv', sep=',', header=True, index=True)
    logi_res.to_csv(rootdir+'/logi_res.csv', sep=',', header=True, index=True)
    rfc_res.to_csv(rootdir+'/rfc_res.csv', sep=',', header=True, index=True)
    xgbc_res.to_csv(rootdir+'/xgbc_res.csv', sep=',', header=True, index=True)
    svc_res.to_csv(rootdir+'/svc_res.csv', sep=',', header=True, index=True)

    