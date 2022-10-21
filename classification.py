import sys
import os
from turtle import forward
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
import torch
import torch.nn as nn
from torch.utils.data import Dataset, dataloader, TensorDataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from sklearn.neural_network import MLPClassifier
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

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
    print("start logi_cv")
    logi = LogisticRegression(n_jobs=10, random_state=RANDOMSEED)
    score = cross_val_score(logi, data, target, cv=5, scoring='roc_auc', n_jobs=10).mean()
    print("end logi_cv")
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
    print("start xgbc_cv")
    xgbc = XGBClassifier(n_estimators=100, n_jobs=10, random_state=RANDOMSEED, learning_rate=0.1)
    score = cross_val_score(xgbc, data, target, cv=5, scoring='roc_auc', n_jobs=10).mean()
    print("end xgbc_cv")
    return score


def rfc_cv(data, target, RANDOMSEED):
    print("start rfc_cv")
    rfc = RandomForestClassifier(n_estimators=100, criterion='entropy', max_features='auto', min_samples_leaf=4, min_samples_split=2, oob_score=True, max_depth=20, n_jobs=10, random_state=RANDOMSEED)
    score = cross_val_score(rfc, data, target, cv=5, scoring='roc_auc', n_jobs=10).mean()
    print("end rfc_cv")
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
    print("start svc_cv")
    svc = SVC(C=0.5, kernel='rbf', gamma='auto', random_state=RANDOMSEED)
    score = cross_val_score(svc, data, target, cv=5, scoring='roc_auc', n_jobs=10).mean()
    print("end svc_cv")
    return score


class MyDataSet():
    def __init__(self, data_tensor, target_tensor=None, transforms=None, target_transforms=None):
        if target_tensor is not None:
            assert data_tensor.size(0) == target_tensor.size(0)

        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
 
        if transforms is None:
            transforms = []
        if target_transforms is None:
            target_transforms = []
 
        if not isinstance(transforms, list):
            transforms = [transforms]
        if not isinstance(target_transforms, list):
            target_transforms = [target_transforms]
 
        self.transforms = transforms
        self.target_transforms = target_transforms
 
    def __getitem__(self, index):
 
        data_tensor = self.data_tensor[index]
        for transform in self.transforms:
            data_tensor = transform(data_tensor)
 
        if self.target_tensor is None:
            return data_tensor
 
        target_tensor = self.target_tensor[index]
        for transform in self.target_transforms:
            target_tensor = transform(target_tensor)
 
        return data_tensor, target_tensor
 
    def __len__(self):
        return self.data_tensor.size(0)


# class TensorsDataset(torch.utils.data.Dataset):
 
#     '''
#     A simple loading dataset - loads the tensor that are passed in input. This is the same as
#     torch.utils.data.TensorDataset except that you can add transformations to your data and target tensor.
#     Target tensor can also be None, in which case it is not returned.
#     '''
 
#     def __init__(self, data_tensor, target_tensor=None, transforms=None, target_transforms=None):
#         if target_tensor is not None:
#             assert data_tensor.size(0) == target_tensor.size(0)
#         self.data_tensor = data_tensor
#         self.target_tensor = target_tensor
 
#         if transforms is None:
#             transforms = []
#         if target_transforms is None:
#             target_transforms = []
 
#         if not isinstance(transforms, list):
#             transforms = [transforms]
#         if not isinstance(target_transforms, list):
#             target_transforms = [target_transforms]
 
#         self.transforms = transforms
#         self.target_transforms = target_transforms
 
#     def __getitem__(self, index):
 
#         data_tensor = self.data_tensor[index]
#         for transform in self.transforms:
#             data_tensor = transform(data_tensor)
 
#         if self.target_tensor is None:
#             return data_tensor
 
#         target_tensor = self.target_tensor[index]
#         for transform in self.target_transforms:
#             target_tensor = transform(target_tensor)
 
#         return data_tensor, target_tensor
 
#     def __len__(self):
#         return self.data_tensor.size(0)
class MyDNN(nn.Module):
    def __init__(self, input_size, hidden1_size=50, hidden2_size=50, output_size=2):
        super(MyDNN, self).__init__()
        # self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden1_size),
            nn.BatchNorm1d(hidden1_size),
            # nn.Dropout(p=0.9),
            nn.ReLU(),
            nn.Linear(hidden1_size, hidden2_size),
            nn.BatchNorm1d(hidden2_size),
            # nn.Dropout(p=0.9),
            nn.ReLU(),
            nn.Linear(hidden2_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train(dataloader, model, loss_fn, optimizer):
    size = (int)(dataloader.dataset.target_tensor.data.sum())*2 # target为1的样本的2倍，与之前sample保持一致
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        # for name in model.state_dict():
        #      print(model.state_dict()[name])
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    # exit()

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0.0, 0.0
    pred_all = None
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            # pred = model(X).squeeze(dim=1)
            # test_loss += loss_fn(pred, y).item()
            # # Expected object of scalar type Long but got scalar type Float for argument #2 
            # print(pred)
            # pred_01 = torch.round(pred)
            # print(pred_01)
            # # pred_01 = pred.
            # correct += (pred_01 == y).type(torch.float).sum().item()
            # if pred_all is None:
            #     pred_all = pred_01.cpu().numpy()
            # else:
            #     pred_all = np.concatenate([pred_all, pred_01.cpu().numpy()])
            pred = model(X)
            # print(pred.argmax(1))

            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            if pred_all is None:
                pred_all = pred.argmax(1).cpu().numpy()
            else:
                pred_all = np.concatenate([pred_all, pred.argmax(1).cpu().numpy()])
    # exit()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.2f}%, Avg loss: {test_loss:>8f} \n")
    return pred_all

def dnn_cv(data, target, RANDOMSEED):
    batch_size = 128
    model = MyDNN(data.shape[1]).to(device)
    print(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    StepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15], gamma=0.1)
    epochs = 20
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, target, test_size=0.2, random_state=RANDOMSEED)
    # train_set = MyDataSet(data=Xtrain, label=Ytrain)
    # test_set = MyDataSet(data=Xtest, label=Ytest)

    # train_loader=DataLoader(train_set,batch_size=batch_size,shuffle=True)
    # test_loader=DataLoader(test_set,batch_size=batch_size,shuffle=False)
    # transform = transforms.Compose([
    #     # transforms.Resize((32, 32)),  # 缩放
    #     # transforms.RandomCrop(32, padding=4),  # 随机裁剪
    #     transforms.ToTensor(),  # 图片转张量，同时归一化0-255 ---》 0-1
    #     # transforms.Normalize(norm_mean, norm_std),  # 标准化均值为0标准差为1
    # ])
    transform = None
    # train_loader = DataLoader(TensorDataset(torch.tensor(np.array(Xtrain)).float(), torch.tensor(np.array(Ytrain)).float()),batch_size=batch_size,shuffle=True, transform=transform)
    # test_loader = DataLoader(TensorDataset(torch.tensor(np.array(Xtest)).float(), torch.tensor(np.array(Ytest)).float()),batch_size=batch_size,shuffle=False, transform=transform)

    train_set = MyDataSet(torch.tensor(np.array(Xtrain)).float(), torch.tensor(np.array(Ytrain)).long(), transforms=transform)
    test_set = MyDataSet(torch.tensor(np.array(Xtest)).float(), torch.tensor(np.array(Ytest)).long(), transforms=transform)

    w = Ytrain.sum() / (len(Ytrain) - Ytrain.sum())
    weights = [w if label == 0 else 1 for data, label in train_set]
    sampler = WeightedRandomSampler(weights, num_samples=(int)(Ytrain.sum()*2), replacement=True)
    # sampler = WeightedRandomSampler(weights, num_samples=27500, replacement=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        # test(test_loader, model, loss_fn)
        train(train_loader, model, loss_fn, optimizer)
        pred = test(test_loader, model, loss_fn)
        StepLR.step()
    print("Done!")
    return roc_auc_score(np.array(Ytest), pred)


def dnn_cv1(data, target, RANDOMSEED):
    print("start dnn_cv")
    dnn = MLPClassifier(hidden_layer_sizes=(50,50),random_state = RANDOMSEED) # 使用两层，每层50个节点
    score = cross_val_score(dnn, data, target, cv=5, scoring='roc_auc', n_jobs=10).mean()
    print("end dnn_cv")
    return score


def clf_cv(clf_name, data, target, RANDOMSEED):
    data=data.loc[:, ~data.columns.duplicated()]
    if clf_name=='rfc':
        return rfc_cv(data, target, RANDOMSEED)
    elif clf_name=='xgbc':
        return xgbc_cv(data, target, RANDOMSEED)
    elif clf_name=='logi':
        return logi_cv(data, target, RANDOMSEED)
    elif clf_name=='svc':
        return svc_cv(data, target, RANDOMSEED)    
    elif clf_name=='dnn':
        return dnn_cv1(data, target, RANDOMSEED)
    

if __name__ == '__main__':
    # data = getBoston()
    data = getMimic()
    # data = getMadelon()
    pprint(rfc_choice_param(data.iloc[:, :-1],data.iloc[:,-1]))
    # data = equidistance_dsct(data)
    # data = entropy_dsct(data)
    # rfc_cv(data, 'Urban')

