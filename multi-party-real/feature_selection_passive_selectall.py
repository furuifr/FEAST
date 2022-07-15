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
"""The Python implementation of the GRPC feature_selection.CalculateScore server."""
from concurrent import futures
import logging
import time

import grpc
import yaml
import feature_selection_pb2
import feature_selection_pb2_grpc

import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split

random.seed(0)

from preprocessing import getData, sampling
from discretization.dsct import dsct
from filter.mutual_info_multi_del import getCrosstab, calc_cond_MI, calc_MI

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

def DfToRepeatedFloat(selectedFeatures, df):
    feature_name_list = list(df.columns)
    for i in range(len(feature_name_list)):
        selectedFeatures.feature_name.append(str(feature_name_list[i]))

    array_list = df.values.tolist()
    for i in range(len(array_list)):
        # print(array_list[i])
        tmp_array = feature_selection_pb2.FloatArray()
        for j in range(len(array_list[i])):
            tmp_array.num.append(array_list[i][j])
        selectedFeatures.array.append(tmp_array)
    return selectedFeatures

class PassiveParty():

    def __init__(self, path):
        # 
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
            self.dataset_name = cfg['dataset_name']
            self.dsct_method = cfg['dsct_method']
            self.dsct_num = cfg['dsct_num']
            self.new_col_num = cfg['new_col_num']
            self.step = cfg['step']
            self.method = cfg['method']
        
        self.model = self.dataset_name+'_'+self.dsct_method+'_'+str(self.step)+'_'+str(self.new_col_num)+'_selectall'
        self.data = getData(self.dataset_name)
        self.data = self.data.iloc[:, 1:]
        self.target = None
        self.data_dsct = dsct(self.dsct_method, self.data[:], self.dsct_num)
        self.cross_feature = []
        self.cond_MI = None
        self.cond_MI_rank = None
        self.isSelected = False

class MFS(feature_selection_pb2_grpc.MFSServicer):

    def SendSamplingRowsAndTarget(self, request, context):
        global passiveParty
        row_list = request.row
        target = request.target
        target = pd.Series(target, name="Target")
        # passiveParty.data = passiveParty.data.iloc[row_list, :].reset_index(drop=True)
        passiveParty.data_dsct = passiveParty.data_dsct.iloc[row_list, :].reset_index(drop=True)
        passiveParty.target = target

        return feature_selection_pb2.Flag(flag = 1)

    def SendCrossFeatures(self, request, context):
        global passiveParty
        df = RepeatedToDf(request)
        passiveParty.cross_feature.append(df)
        cf = passiveParty.cross_feature[0]
        for i in range(1, len(passiveParty.cross_feature)):
            cf = pd.concat([cf, passiveParty.cross_feature[i]], axis=1)
        tmp_cond_MI, tmp_cond_MI_rank = calc_cond_MI(cf, passiveParty.data_dsct, passiveParty.target, passiveParty.method)
        passiveParty.cond_MI = tmp_cond_MI
        passiveParty.cond_MI_rank = tmp_cond_MI_rank
        print(tmp_cond_MI)
        return feature_selection_pb2.FeatureScores(score = tmp_cond_MI)

    def GetMI(self, request, context):
        global passiveParty
        tmp_cond_MI, tmp_cond_MI_rank = calc_MI(passiveParty.data_dsct, passiveParty.target)
        passiveParty.cond_MI = tmp_cond_MI
        passiveParty.cond_MI_rank = tmp_cond_MI_rank
        print(tmp_cond_MI)
        return feature_selection_pb2.FeatureScores(score = tmp_cond_MI)

    def GetCrossFeatures(self, request, context):
        global passiveParty
        selected_num = request.selected_num
        passiveParty.cond_MI = passiveParty.cond_MI[:selected_num]
        passiveParty.cond_MI_rank = passiveParty.cond_MI_rank[:selected_num]
        passiveParty.data = passiveParty.data.iloc[:, passiveParty.cond_MI_rank]
        passiveParty.data_dsct = passiveParty.data_dsct.iloc[:, passiveParty.cond_MI_rank]

        crossFeatures = feature_selection_pb2.CrossFeatures()
        cross_feature = getCrosstab(passiveParty.data_dsct, passiveParty.target, passiveParty.step, passiveParty.model+passiveParty.method+str(0))
        crossFeatures = DfToRepeated(crossFeatures, cross_feature)
        
        print(list(passiveParty.data.columns))

        return feature_selection_pb2.CrossFeatures(feature_name = crossFeatures.feature_name, array = crossFeatures.array)

    def SendDelFeatures(self, request, context):
        global passiveParty
        del_num = request.del_num
        if del_num<len(passiveParty.cond_MI_rank):
            passiveParty.cond_MI_rank = passiveParty.cond_MI_rank[:-del_num]            
            passiveParty.data = passiveParty.data.iloc[:, passiveParty.cond_MI_rank]
            passiveParty.data_dsct = passiveParty.data_dsct.iloc[:, passiveParty.cond_MI_rank]
        return feature_selection_pb2.Flag(flag = 1)

    def GetSelectedFeatures(self, request, context):
        global passiveParty

        selectedFeatures = feature_selection_pb2.SelectedFeatures()
        selectedFeatures = DfToRepeatedFloat(selectedFeatures, passiveParty.data)

        return feature_selection_pb2.SelectedFeatures(feature_name = selectedFeatures.feature_name, array = selectedFeatures.array)

def serve():
    global passiveParty
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    feature_selection_pb2_grpc.add_MFSServicer_to_server(MFS(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("server start...")
    while 1:
        time.sleep(10)



if __name__ == '__main__':
    logging.basicConfig()
    passiveParty = PassiveParty(path='passive_param_selectall.yaml')
    serve()
