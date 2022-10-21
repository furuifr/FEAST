import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
import yaml
from preprocessing import getData
from discretization.dsct import dsct
from filter.mutual_info_single import calc_cond_MI, calc_MI
from classification import clf_cv
from pprint import pprint
import random
import warnings;
warnings.filterwarnings(action='once')
warnings.filterwarnings("ignore")

def featureSelection(new_col_num, data, data_dsct, target, method, dataset_name, dsct_method, all_columns, RANDOMSEED, subdir):
    
    acc_res_fl_ratio_logi = []
    acc_res_fl_ratio_svc = []
    acc_res_fl_ratio_rfc = []
    acc_res_fl_ratio_xgbc = []
    acc_res_fl_ratio_dnn = []
    data_selected, data_selected_dsct = pd.DataFrame(), pd.DataFrame()

    for i in range(new_col_num):
        model = dataset_name+'_'+dsct_method+'_'+str(i+1)
        print(i)
        if i==0:
            tmp_cond_MI, tmp_cond_MI_rank = calc_MI(data_dsct, target)
            select_col = tmp_cond_MI_rank[0]
            data_selected = data.iloc[:, select_col].to_frame()
            data_selected_dsct = data_dsct.iloc[:, select_col].to_frame()
            data.drop(list(data.columns[[select_col]]), axis=1, inplace=True)
            data_dsct.drop(list(data_dsct.columns[[select_col]]), axis=1, inplace=True)
            
        else:
            tmp_cond_MI, tmp_cond_MI_rank = calc_cond_MI(data_selected_dsct, data_dsct, target, method)
            select_col = tmp_cond_MI_rank[0]
            data_selected = pd.concat([data_selected, data.iloc[:, select_col].to_frame()], axis=1)
            data_selected_dsct = pd.concat([data_selected_dsct, data_dsct.iloc[:, select_col].to_frame()], axis=1)
            data.drop(list(data.columns[[select_col]]), axis=1, inplace=True)
            data_dsct.drop(list(data_dsct.columns[[select_col]]), axis=1, inplace=True)
        
        if i>35:
            acc_res_fl_ratio_logi.append(clf_cv('logi', data_selected.copy(), target, RANDOMSEED))
            # acc_res_fl_ratio_svc.append(clf_cv('svc', data_selected.copy(), target, RANDOMSEED))
            acc_res_fl_ratio_rfc.append(clf_cv('rfc', data_selected.copy(), target, RANDOMSEED))
            acc_res_fl_ratio_xgbc.append(clf_cv('xgbc', data_selected.copy(), target, RANDOMSEED))
            acc_res_fl_ratio_dnn.append(clf_cv('dnn', data_selected.copy(), target, RANDOMSEED))

            print(list(data_selected.columns))
            # print(list(map(lambda x: all_columns.index(x), list(data_sum_fl_ratio.columns))))
            # print(acc_res_fl_ratio)

            print(acc_res_fl_ratio_logi)
            print(acc_res_fl_ratio_svc)
            print(acc_res_fl_ratio_rfc)
            print(acc_res_fl_ratio_xgbc)
            print(acc_res_fl_ratio_dnn)

            with open(subdir+'/logi_'+model+method+'.txt', 'w') as f:
                f.write(str(list(data_selected.columns)))
                f.write('\r\n')
                f.write(str(list(map(lambda x: all_columns.index(x), list(data_selected.columns)))))
                f.write('\r\n')
                f.write(str(acc_res_fl_ratio_logi))
                f.write('\r\n')

            # with open(subdir+'/svc_'+model+method+'.txt', 'w') as f:
            #     f.write(str(list(data_selected.columns)))
            #     f.write('\r\n')
            #     f.write(str(list(map(lambda x: all_columns.index(x), list(data_selected.columns)))))
            #     f.write('\r\n')
            #     f.write(str(acc_res_fl_ratio_svc))
            #     f.write('\r\n')

            with open(subdir+'/rfc_'+model+method+'.txt', 'w') as f:
                f.write(str(list(data_selected.columns)))
                f.write('\r\n')
                f.write(str(list(map(lambda x: all_columns.index(x), list(data_selected.columns)))))
                f.write('\r\n')
                f.write(str(acc_res_fl_ratio_rfc))
                f.write('\r\n')

            with open(subdir+'/xgbc_'+model+method+'.txt', 'w') as f:
                f.write(str(list(data_selected.columns)))
                f.write('\r\n')
                f.write(str(list(map(lambda x: all_columns.index(x), list(data_selected.columns)))))
                f.write('\r\n')
                f.write(str(acc_res_fl_ratio_xgbc))
                f.write('\r\n')

            with open(subdir+'/dnn_'+model+method+'.txt', 'w') as f:
                f.write(str(list(data_selected.columns)))
                f.write('\r\n')
                f.write(str(list(map(lambda x: all_columns.index(x), list(data_selected.columns)))))
                f.write('\r\n')
                f.write(str(acc_res_fl_ratio_dnn))
                f.write('\r\n')
