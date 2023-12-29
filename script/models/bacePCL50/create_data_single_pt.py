# -*- coding: iso-8859-1 -*-
import pandas as pd
import numpy as np
import os
import json,pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils import *
from utils_smiecfp import * 
from data_gen_modify import *

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    return c_size, features, edge_index

def getProcessData(label):
    if label == 'train':
        allFoldPath = '../../../dataSour/bacePLC50/dataTrain.dat'
    else:
        allFoldPath = '../../../dataSour/bacePLC50/dataTest.dat'
    vocab_path = '../../../dataSour/bacePLC50/smiles_char_dict.pkl'
    with open(vocab_path, 'rb') as f:
        smilesVoc = pickle.load(f)
    allData = getData(allFoldPath)
    smiles = []
    encodedSmi = []
    ecfp = []
    properties = []
    for index, item in enumerate(allData):
        smiles.append(item[0])
        enSmi = item[1].split(',')
        enSmi = [int(float(val)) for val in enSmi]
        encodedSmi.append(enSmi)
        ep = item[2].split(',')
        ep = [int(val) for val in ep]
        ecfp.append(ep)
        properties.append(item[3])
    smi_to_graph = {}
    resultSmi = []
    resultEp = []
    resultY = []
    count = 0
    for smi, enSmi, ep, y in zip(smiles, encodedSmi, ecfp,properties):
        # print('{%s/%s}' %(index, len(ligand)))
        c_size, features, edge_index = smile_to_graph(smi)
        if edge_index == []:
            continue
        smi_to_graph[count] = smile_to_graph(smi)
        resultSmi.append(enSmi)
        resultEp.append(ep)
        resultY.append(y)
        count = count + 1
    return resultSmi, resultEp, resultY, smi_to_graph


processed_data_file_train = '../../../dataSour/bacePLC50/processed/data_train.pt'
processed_data_file_test = '../../../dataSour/bacePLC50/processed/data_test.pt'
if (not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test)):
    train_enSmi, train_ep, train_pro, train_smiGraph = getProcessData('train')
    test_enSmi, test_ep, test_pro, test_smiGraph = getProcessData('test')
    train_data = formDataset_Single(root='../../../dataSour/bacePLC50/', dataset='data_train', encodedSmi=train_enSmi, ecfp=train_ep, y=train_pro, smile_graph=train_smiGraph)
    test_data = formDataset_Single(root='../../../dataSour/bacePLC50/', dataset='data_test', encodedSmi=test_enSmi, ecfp=test_ep, y=test_pro, smile_graph=test_smiGraph)
    print('preparing data_train.pt in pytorch format!')
    print('preparing data_test.pt in pytorch format!')
else:
    print('preparing data_train.pt is already created!')
    print('preparing data_test.pt is already created!')
