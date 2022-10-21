from fileinput import filename
import imp
import os
import numpy as np
import pandas as pd
import re

def subdir(filepath):
    pathDir =  os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join('%s\%s' % (filepath, allDir))
        if os.path.isfile(child):
            if os.path.splitext(child)[-1] != '.txt':
                continue
            readFile(child)
            print(child)
            continue
        subdir(child)
    
def readFile(filenames):
    name_list = filenames.split("_")
    type = name_list[0].split("\\")[-1]
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
                    logi_res.loc[int(name_list[4])][name_list[-2]] = auc
                elif type == 'rfc':
                    rfc_res.loc[int(name_list[4])][name_list[-2]] = auc
                elif type == 'svc':
                    svc_res.loc[int(name_list[4])][name_list[-2]] = auc
                elif type == 'xgbc':
                    xgbc_res.loc[int(name_list[4])][name_list[-2]] = auc
        
if __name__ == "__main__":
    rootdir = 'D:\\付瑞\\OneDrive\\res\\New Folder\\b10m3'
    # columns = {'CFR', 'chi2', 'CIFE', 'CMIM', 'pro10m', 'pro10', 'f', 'IV', 'JMI', 'MRMR', 'info', 'SU'}
    columns = {'FEAST'}
    index = [(i+1) for i in range(30)]
    logi_res = pd.DataFrame(columns=columns, index=index)
    rfc_res = pd.DataFrame(columns=columns, index=index)
    xgbc_res = pd.DataFrame(columns=columns, index=index)
    svc_res = pd.DataFrame(columns=columns, index=index)
    subdir(rootdir)
    logi_res.to_csv('logi_res.csv', sep=',', header=True, index=True)
    rfc_res.to_csv('rfc_res.csv', sep=',', header=True, index=True)
    xgbc_res.to_csv('xgbc_res.csv', sep=',', header=True, index=True)
    svc_res.to_csv('svc_res.csv', sep=',', header=True, index=True)

    