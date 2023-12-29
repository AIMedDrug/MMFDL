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
        allFoldPath = '../../../dataSour/refined/dataTrain.dat'
    else:
        allFoldPath = '../../../dataSour/refined/dataTest.dat'
    allData = getData(allFoldPath)
    encodedSmi = []
    ecfp = []
    log = []
    smi_to_graph_lig = {}
    for index, item in enumerate(allData):
        smi_lig = item[0]
        smi_to_graph_lig[index] = smile_to_graph(smi_lig)
        enSmi = item[2].split(',')
        enSmi = [int(float(val)) for val in enSmi]
        encodedSmi.append(enSmi)
        ep = item[3].split(',')
        ep = [int(val) for val in ep]
        ecfp.append(ep)
        log.append(item[4])
    return encodedSmi, ecfp, log, smi_to_graph_lig




processed_data_file_train = '../../../dataSour/refined/processed/data_train.pt'
processed_data_file_test = '../../../dataSour/refined/processed/data_test.pt'
if (not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test)):

    train_encodedSmi, train_ecfp, train_y, train_smiGraph = getProcessData('train')
    test_encodedSmi, test_ecfp, test_y, test_smiGraph = getProcessData('test')

    train_data = formDataset(root='../../../dataSour/refined', dataset='data_train', encodedSmi=train_encodedSmi, ecfp=train_ecfp, y=train_y, smile_graph=train_smiGraph)
    test_data = formDataset(root='../../../dataSour/refined', dataset='data_test', encodedSmi=test_encodedSmi, ecfp=test_ecfp, y=test_y, smile_graph=test_smiGraph)
    print('preparing data_train.pt in pytorch format!')
else:
    print('preparing data_train.pt is already created!')