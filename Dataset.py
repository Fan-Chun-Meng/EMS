import os
import re
import math
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
import numpy as np

dic_train = {}
dic_test = {}
def get_float_label(file_name):
    e = str(file_name).split("\t")[1].split("\n")[0]
    return float(e)


def read_list_from_disk(path):
    folderList = os.listdir(path)

    paths = []
    for i in range(len(folderList)):
        path1 = os.path.join(path, folderList[i])
        paths.append(path1)
    filenames = []
    for i in range(len(paths)):
        for root, dirs, files in os.walk(paths[i]):
            for j in range(len(files)):
                filePath = os.path.join(paths[i], files[j])
                if 'BHZ' in filePath:
                    filenames.append(filePath)
    return filenames

def read_list_from_disk_test(path):
    f = open("shiyan/zonghe.txt","r")
    filenames = []

    for i in f.readlines():
        filenames.append(i.split("\t")[0])
    return filenames

def read_zq(data_list, path):
    mul = int(path.split("zq-")[1][0])
    for i in range(3000):
        data_list[10000+i] = str(float(data_list[10000+i]) + float(data_list[100*(mul-1)+i]))+"\n"
    return data_list

class MyDataset(Dataset):

    def __init__(self, dataPath, is_test):

        data = open(dataPath, "r", encoding="utf-8").readlines()
        self.data = data
        self.is_test = is_test

    def __getitem__(self, index):

        path = self.data[index]
        dataPath = path.split("\t")[0]
        try:
            label = float(path.split("\t")[1])*10
        except:
            print(path)

        try:
            npfile = "E:/bslw实验/2.震级回归/数据/npy文件/0/train-final/"+dataPath.split("/")[4]+"_"+dataPath.split("/")[5].split(".txt")[0]+".npy"
        except:
            print(dataPath)
        dataSet = np.load(npfile)
        dataSet = dataSet[:,:3000]
        dataSet_Nor = (dataSet-np.min(dataSet, axis=1, keepdims=True))/(np.max(dataSet, axis=1, keepdims=True)-np.min(dataSet, axis=1, keepdims=True))
        return torch.tensor(dataSet), dataSet_Nor, torch.tensor(float(label))

    def __len__(self):
        return len(self.data)

