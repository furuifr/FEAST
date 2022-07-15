# Copyright 2015 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Python implementation of the GRPC feature_selection.CalculateScore client."""

from __future__ import print_function

import logging
from os import name

import grpc
from numpy.lib.function_base import append
from pandas.core.frame import DataFrame
from classification import clf_cv
import feature_selection_pb2
import feature_selection_pb2_grpc

import numpy as np
import pandas as pd
import yaml
import random
import time
from sklearn.model_selection import train_test_split
from multiprocessing import Process, Manager

random.seed(0)
from preprocessing import getData, sampling, getSamplingRow
from discretization.dsct import dsct
from filter.mutual_info_multi_del import calc_MI, getCrosstab, calc_cond_MI

class ActiveParty():

    def __init__(self, path):
        # 
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
            self.dataset_name = cfg['dataset_name']
            self.target_name = cfg['target']
            self.dsct_method = cfg['dsct_method']
            self.dsct_num = cfg['dsct_num']
            self.new_col_num = cfg['new_col_num']
            self.step = cfg['step']
            self.is_sampling = cfg['is_sampling']
            self.sampling_method = cfg['sampling_method']
            self.method = cfg['method']
            self.party_num = cfg['party_num']

        self.model = self.dataset_name+'_'+self.dsct_method+'_'+str(self.step)+'_'+str(self.new_col_num)+'_selectall'
        self.data = getData(self.dataset_name)
        self.target = self.data[self.target_name]
        self.target.rename('Target', inplace=True)
        self.data = self.data.iloc[:, 2:].reset_index(drop=True)
        self.target_ori = self.target[:]
        self.cross_feature = []
        self.cond_MI = None
        self.cond_MI_rank = None

        self.passive_party = [False for _ in range(self.party_num)]

        # self.sampling_row 为分层采样后的行数，这里还需要改
        if self.is_sampling:
            self.sampling_row = getSamplingRow(self.data, self.target, self.sampling_method, self.dsct_num, self.step)
        else:
            self.sampling_row = list(range(len(self.data)))

        self.data_cut = self.data.iloc[self.sampling_row, :].reset_index(drop=True)
        self.data_dsct = dsct(self.dsct_method, self.data_cut[:], self.dsct_num)
        # self.data_dsct = self.data_dsct.iloc[self.sampling_row, :].reset_index(drop=True)
        self.target = self.target.iloc[self.sampling_row].reset_index(drop=True)


    # def getScoreInA(self):
    #     if len(self.cross_feature)==0:
    #         self.cond_MI, self.cond_MI_rank = calc_MI(self.data_dsct, self.target)
    #     else:
    #         cf = self.cross_feature[0]
    #         for i in range(1, len(self.cross_feature)):
    #             cf = pd.concat([cf, self.cross_feature[i]], axis=1)
    #         self.cond_MI, self.cond_MI_rank = calc_cond_MI(cf, self.data_dsct, self.target, self.method)
    #     return feature_selection_pb2.FeatureScores(score = self.cond_MI)

def DfToRepeated(crossFeatures, df):
    feature_name_list = list(df.columns)
    for i in range(len(feature_name_list)):
        crossFeatures.feature_name.append(str(feature_name_list[i]))

    array_list = df.values.tolist()
    for i in range(len(array_list)):
        # print(array_list[i])
        tmp_array = feature_selection_pb2.Array()
        for j in range(len(array_list[i])):
            tmp_array.num.append(array_list[i][j])
        crossFeatures.array.append(tmp_array)
    return crossFeatures

def RepeatedToDf(response):
    feature_name = response.feature_name
    df_list = []
    for arr in response.array:
        df_list.append(arr.num)
    df = pd.DataFrame(data=df_list, columns=feature_name)
    return df

def SamplingRowsAndTargetInA(path, return_dict, id):
    global activeParty
    with grpc.insecure_channel(path) as channel:
        stub = feature_selection_pb2_grpc.MFSStub(channel)
        samplingRowsAndTarget = feature_selection_pb2.SamplingRowsAndTarget()
        for i in range(len(activeParty.sampling_row)):
            samplingRowsAndTarget.row.append(activeParty.sampling_row[i])
        target_list = activeParty.target.tolist()    
        for i in range(len(target_list)):
            samplingRowsAndTarget.target.append(target_list[i])
        # print(samplingRowsAndTarget.row)
        # print(target_list)
        response = stub.SendSamplingRowsAndTarget(feature_selection_pb2.SamplingRowsAndTarget(row = samplingRowsAndTarget.row, target = target_list))
        return_dict[path] = response.flag

def SendCrossFeaturesInA(path, return_dict, id):
    global activeParty
    return_dict['count'] += 1
    if id==0:
        if len(activeParty.cross_feature)==0:
            tmp_cond_MI, tmp_cond_MI_rank = calc_MI(activeParty.data_dsct, activeParty.target)
        else:
            cf = activeParty.cross_feature[0]
            for i in range(1, len(activeParty.cross_feature)):
                cf = pd.concat([cf, activeParty.cross_feature[i]], axis=1)
            tmp_cond_MI, tmp_cond_MI_rank = calc_cond_MI(cf, activeParty.data_dsct, activeParty.target, activeParty.method)
        return_dict['cond_MI'] = tmp_cond_MI
        return_dict['cond_MI_rank'] = tmp_cond_MI_rank
        return_dict[path] = feature_selection_pb2.FeatureScores(score = tmp_cond_MI)
    else:
        with grpc.insecure_channel(path) as channel:
            stub = feature_selection_pb2_grpc.MFSStub(channel)
            if len(activeParty.cross_feature)==0:
                response = stub.GetMI(feature_selection_pb2.Flag(flag = 1))
                return_dict[path] = response
            else:
                crossFeatures = feature_selection_pb2.CrossFeatures()
                crossFeatures = DfToRepeated(crossFeatures, activeParty.cross_feature[-1])
                response = stub.SendCrossFeatures(feature_selection_pb2.CrossFeatures(feature_name = crossFeatures.feature_name, array = crossFeatures.array))
                return_dict[path] = response

def GetCrossFeaturesInA(path, selected_num, next_party):
    global activeParty
    if next_party==0:
        activeParty.cond_MI = activeParty.cond_MI[:selected_num]
        activeParty.cond_MI_rank = activeParty.cond_MI_rank[:selected_num]
        activeParty.data = activeParty.data.iloc[:, activeParty.cond_MI_rank]
        activeParty.data_dsct = activeParty.data_dsct.iloc[:, activeParty.cond_MI_rank]
        df = getCrosstab(activeParty.data_dsct, activeParty.target, activeParty.step, activeParty.model+activeParty.method+str(0))
        activeParty.cross_feature.append(df)
        print(list(activeParty.data.columns))
    else:
        with grpc.insecure_channel(path) as channel:
            stub = feature_selection_pb2_grpc.MFSStub(channel)
            response = stub.GetCrossFeatures(feature_selection_pb2.SelectedNum(selected_num = selected_num))
            df = RepeatedToDf(response)
            activeParty.cross_feature.append(df)
    activeParty.passive_party[next_party] = True
    activeParty.new_col_num -= selected_num # 删除已选的个数

    # return_dict[str(id)] = response

def SendDelFeaturesInA(path, del_num, return_dict, id):
    global activeParty
    if id == 0:
        return_dict[path] = 1
    else:
        with grpc.insecure_channel(path) as channel:
            stub = feature_selection_pb2_grpc.MFSStub(channel)
            if del_num==0:
                return_dict[path] = 1
            else:
                response = stub.SendDelFeatures(feature_selection_pb2.DelNum(del_num = del_num))
                return_dict[path] = response.flag

def GetSelectedFeaturesInA(path, return_dict, id):
    with grpc.insecure_channel(path) as channel:
        stub = feature_selection_pb2_grpc.MFSStub(channel)
        response = stub.GetSelectedFeatures(feature_selection_pb2.Flag(flag = True))
        df = RepeatedToDf(response)
        return_dict[path] = df

def SelectNextParty(mi_score, path):
    global activeParty
    cond_MI_other_party_all = []
    cond_MI_other_party_list = [] # 第一个数为个数，第二个数为MI的总和
    other_party_delete_list = []

    for i in range(activeParty.party_num):
        cond_MI_other_party_list.append([0, 0]) # 第一个数为个数，第二个数为MI的总和
        other_party_delete_list.append(0) # 要删除的个数
        if mi_score[path[i]] == None:
            continue
        for j in range(len(mi_score[path[i]].score)):
            cond_MI_other_party_all.append((i, mi_score[path[i]].score[j]))

    cond_MI_other_party_all = sorted(cond_MI_other_party_all, key=lambda x: -x[1])
    
    delete_num = int((len(cond_MI_other_party_all)-activeParty.new_col_num) * 0.2)
    # 遍历后delete_num个features
    for j in range(len(cond_MI_other_party_all)-delete_num, len(cond_MI_other_party_all)):
        other_party_delete_list[cond_MI_other_party_all[j][0]] += 1

    # 这里先选总和最大的，之后再调整
    for j in range(activeParty.new_col_num):
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

    return next_active, next_feature_num, other_party_delete_list

def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code. 10.108.17.26
    global activeParty, start
    manager = Manager()

    # path = ['47.94.231.32', '47.93.176.80:50051', '123.56.99.44:50052', '47.94.46.217:50053']
    path = ['47.94.231.32', '47.93.176.80:50051']

    jobs = []
    return_dict = manager.dict()
    # 传递sampling的行和target
    for i in range(1, activeParty.party_num):
        jobs.append(Process(target=SamplingRowsAndTargetInA, args=(path[i], return_dict, i)))
        jobs[-1].start()
    print('before join')
    for job in jobs:
        job.join()
    print('after join')
    for i in range(1, activeParty.party_num):
        print(return_dict[path[i]])

    for round in range(activeParty.party_num):

        if activeParty.new_col_num==0:
            break

        jobs = []
        return_dict = manager.dict()
        return_dict['count'] = 0
        # 传递cross feature并得到score
        for i in range(activeParty.party_num):
            if activeParty.passive_party[i] == False:
                jobs.append(Process(target=SendCrossFeaturesInA, args=(path[i], return_dict, i)))
                jobs[-1].start()
            else:
                return_dict[path[i]] = None
        print('before join')
        for job in jobs:
            job.join()
        print('after join')
        for i in range(activeParty.party_num):
            print(return_dict[path[i]])
        if activeParty.passive_party[0] == False:
            activeParty.cond_MI = return_dict['cond_MI']
            activeParty.cond_MI_rank = return_dict['cond_MI_rank']

        # 计算得到next party，以及选中的特征个数和所有party要删除的特征个数
        next_party, selected_num, del_list = SelectNextParty(return_dict, path)
        GetCrossFeaturesInA(path[next_party], selected_num, next_party)

        jobs = []
        return_dict = manager.dict()
        # 传递所有party要删除的特征个数
        for i in range(activeParty.party_num):
            if activeParty.passive_party[i] == False:
                if i == 0:
                    if del_list[i]!=0 and del_list[i]<len(activeParty.cond_MI_rank):
                        activeParty.cond_MI_rank = activeParty.cond_MI_rank[:-del_list[i]]            
                        activeParty.data = activeParty.data.iloc[:, activeParty.cond_MI_rank]
                        activeParty.data_dsct = activeParty.data_dsct.iloc[:, activeParty.cond_MI_rank]
                jobs.append(Process(target=SendDelFeaturesInA, args=(path[i], del_list[i], return_dict, i)))
                jobs[-1].start()
            else:
                return_dict[path[i]] = None
        print('before join')
        for job in jobs:
            job.join()
        print('after join')
        for i in range(activeParty.party_num):
            print(return_dict[path[i]])

    end = time.time()
    rtime = end - start
    print(rtime)
    
    jobs = []
    return_dict = manager.dict()
    # 获取选择的特征
    for i in range(1, activeParty.party_num):
        if activeParty.passive_party[i] == True:
            jobs.append(Process(target=GetSelectedFeaturesInA, args=(path[i], return_dict, i)))
            jobs[-1].start()
        else:
            return_dict[path[i]] = None
    print('before join')
    for job in jobs:
        job.join()
    print('after join')
    for i in range(1, activeParty.party_num):
        print(return_dict[path[i]])

    # 检查传递回来的数据
    df = activeParty.data
    for i in range(1, activeParty.party_num):
        if return_dict[path[i]] is not None:
            df = pd.concat([df, return_dict[path[i]]], axis=1)
    acc_res = []
    # acc_res.append(clf_cv('logi', df.copy().T.drop_duplicates().T, activeParty.target_ori))
    acc_res.append(clf_cv('svc', df.copy().T.drop_duplicates().reset_index(drop=True).T, activeParty.target_ori))
    acc_res.append(clf_cv('rfc', df.copy().T.drop_duplicates().reset_index(drop=True).T, activeParty.target_ori))
    acc_res.append(clf_cv('xgbc', df.copy().T.drop_duplicates().reset_index(drop=True).T, activeParty.target_ori))
    print(acc_res)

    with open(activeParty.model+activeParty.method+'.txt', 'w') as f:
        f.write(str(list(df.columns)))
        f.write('\r\n')
        f.write(str(acc_res))
        f.write('\r\n')
        f.write(str(rtime))
        f.write('\r\n')


if __name__ == '__main__':
    logging.basicConfig()
    start = time.time()
    activeParty = ActiveParty(path='active_param_selectall.yaml')
    run()
