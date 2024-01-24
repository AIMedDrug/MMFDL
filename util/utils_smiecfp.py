# -*- coding: iso-8859-1 -*-
import numpy as np
import re
import torch
from torch.autograd import Variable
import pandas as pd
import random
from data_gen_modify import line2voc_arr
from collections import Counter


def tokenizer(smile):
    "Tokenizes SMILES string"
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|_|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regezz = re.compile(pattern)
    tokens = [token for token in regezz.findall(smile)]
    assert smile == ''.join(tokens), ("{} could not be joined".format(smile))
    return tokens



def getSmiLen(smiles, letters):
    sequence = [line2voc_arr(line,letters) for line in smiles]
    vectorized_seqs = [sl[0] for sl in sequence]
    seq_len = []
    for vector in vectorized_seqs:
        seq_len.append(len(vector))
    return max(seq_len)

def replace_halogen(string):
    """Regex to replace Br and Cl with single letters"""
    br = re.compile('Br')
    cl = re.compile('Cl')
    string = br.sub('R', string)
    string = cl.sub('L', string)
    return string

def getLetters(path):
    with open(path, 'r') as f:
        chars = f.read().split()
    return chars

def getData(foldPath):
    allData = []
    with open(foldPath, "r") as dat_file:
        for line in dat_file:

            line = line.strip() 
            elements = line.split(" ") 
            allData.append(elements)
    return allData

def getSplitDataSet(foldPath, valid_rate):
    dataSet = getData(foldPath)
    random.shuffle(dataSet)
    split_point = int(len(dataSet) * (1-valid_rate))
    train_data = dataSet[:split_point]
    val_data = dataSet[split_point:]
    return train_data, val_data

def getDataSet_train(foldPath, valid_rate):
    dataSet = getData(foldPath)
    random.shuffle(dataSet)
    split_point = int(len(dataSet) * (1-valid_rate))
    trainSet = dataSet[:split_point]
    valSet = dataSet[split_point:]
    return trainSet, valSet

def getDataSet_test(foldPath):
    dataSet = getData(foldPath)
    return dataSet

def getVocSmi_single(smileVocPath, dataFoldPath, label):
    smilesVoc = getLetters(smileVocPath)
    allData = getData(dataFoldPath)
    if label == 0:
        ligand = []
        pocket = []
        for index, item in enumerate(allData):
            ligand.append(item[1])
            pocket.append(item[2])
        
        max_ligand_len = getSmiLen(ligand, smilesVoc)
        max_pocket_len = getSmiLen(pocket, smilesVoc)
        return smilesVoc, max_ligand_len, max_pocket_len
    elif label == 1:
        smiles = []
        for index, item in enumerate(allData):
            smiles.append(item[0])
        max_smiles_len = getSmiLen(smiles, smilesVoc)
        return smilesVoc, max_smiles_len
    
def getInput_mask(data):
    mask_array = np.zeros((len(data), data.shape[1]), dtype=np.int)
    for idx, item in enumerate(data):
        temp = np.zeros(data.shape[1])
        for i_idx, ele in enumerate(item):
            if ele > 0:
                temp[i_idx] = 1
        mask_array[idx] = torch.LongTensor(temp)
    return mask_array