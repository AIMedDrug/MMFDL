# -*- coding: iso-8859-1 -*-
import numpy as np
import re
import torch
from torch.autograd import Variable
import pandas as pd
import random
from data_gen_modify import line2voc_arr




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

# ÖÆ×÷×ÖµäµÄ£¬Ê²Ã´¹æÔò£¬Ó¦¸ÃÊÇÕýÔò±í´ïÊ½
def construct_vocabulary(smiles_list, savePath):
    """Returns all the characters present in a SMILES file.
       Uses regex to find characters/tokens of the format '[x]'."""
    add_chars = set()
    for i, smiles in enumerate(smiles_list):
        regex = '(\[[^\[\]]{1,10}\])'
        smiles = replace_halogen(smiles)
        char_list = re.split(regex, smiles)
        for char in char_list:
            if char.startswith('['):
                add_chars.add(char)
            else:
                chars = [unit for unit in char]
                [add_chars.add(unit) for unit in chars]

    print("Number of characters: {}".format(len(add_chars)))
    with open(savePath, 'w') as f:
        f.write('<pad>' + "\n")
        for char in add_chars:
            f.write(char + "\n")
    return add_chars


def getLetters(path):
    with open(path, 'r') as f:
        chars = f.read().split()
    return chars



'''
    construct dataset
'''
def getData(foldPath):
    allData = []
    # ´ò¿ª .dat ÎÄ¼þÒÔ¶ÁÈ¡Ä£Ê½
    with open(foldPath, "r") as dat_file:
        # ÖðÐÐ¶ÁÈ¡ÎÄ¼þÄÚÈÝ
        for line in dat_file:
            # È¥³ý»»ÐÐ·û²¢ÒÔ¿Õ¸ñ·Ö¸î×Ö·û
            line = line.strip()  # È¥³ý»»ÐÐ·ûºÍ¿Õ°××Ö·û
            elements = line.split(" ")  # ÒÔ¿Õ¸ñ·Ö¸î×Ö·û
            allData.append(elements)
    # print(allData)
    # exit()
    # with open(foldPath, 'r') as f:
    #     data = f.read().strip().split('\n')
    # dataList = [x.strip().split(',') for x in data]
   
    return allData

def getSplitDataSet(foldPath, valid_rate):
    dataSet = getData(foldPath)
    # Ëæ»ú´òÂÒÊý¾Ý¼¯
    random.shuffle(dataSet)
    # ¼ÆËã·Ö¸îµã
    split_point = int(len(dataSet) * (1-valid_rate))
    
    # »®·ÖÊý¾Ý¼¯£¬Ç°90%ÓÃÓÚÑµÁ·£¬ºó10%ÓÃÓÚ²âÊÔ
    train_data = dataSet[:split_point]
    val_data = dataSet[split_point:]
    return train_data, val_data


def getVocSmi():
    smileVocPath = '../../dataSour/teacher/smileAllVoc.voc'
    smilesVoc = getLetters(smileVocPath)

    allFoldPath = '../../data_ecfp/new/allData_head.dat'
    allData = getData(allFoldPath)
    ligand = []
    pocket = []
    for index, item in enumerate(allData):
        if index == 0:
            continue
        ligand.append(item[1])
        pocket.append(item[2])
    
    max_ligand_len = getSmiLen(ligand, smilesVoc)
    max_pocket_len = getSmiLen(pocket, smilesVoc)
    return smilesVoc, max_ligand_len, max_pocket_len


def getInput_mask(data):
    mask_array = np.zeros((len(data), data.shape[1]), dtype=np.int)
    for idx, item in enumerate(data):
        temp = np.zeros(data.shape[1])
        for i_idx, ele in enumerate(item):
            if ele > 0:
                temp[i_idx] = 1
        mask_array[idx] = torch.LongTensor(temp)
    return mask_array
    
# if __name__ == '__main__':
#     ligandList = pd.read_csv('/home/bioinfor3/Lxh/transformerCPI/myTrCPI/dataSour/teacher/smiles_refined_set.csv')['ligand'].tolist()
#     pocketList = pd.read_csv('/home/bioinfor3/Lxh/transformerCPI/myTrCPI/dataSour/teacher/smiles_refined_set.csv')['pocket'].tolist()
#     smile_List = ligandList + pocketList
#     savePath = '/home/bioinfor3/Lxh/transformerCPI/myTrCPI/dataSour/teacher/smileAllVoc.voc'
#     construct_vocabulary(smile_List, savePath)
    # foldPath = '/home/bioinfor3/Lxh/transformerCPI/myTrCPI/data/dataTrain.txt'
    # getDataSet(foldPath)
