import os
import numpy as np
import pickle
import pandas as pd
import torch
from torch.autograd import Variable
import re
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from utils_smiecfp import *
import warnings

# 禁用RDKit的警告
rdkit.RDLogger.DisableLog('rdApp.*')


def tokenizer(smile):
    "Tokenizes SMILES string"
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|_|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regezz = re.compile(pattern)
    tokens = [token for token in regezz.findall(smile)]
    assert smile == ''.join(tokens), ("{} could not be joined".format(smile))
    return tokens

# 在处理分子之前，将RDKit的日志级别设置为ERROR
rdkit.RDLogger.logger().setLevel(rdkit.RDLogger.ERROR)

def create_variable(tensor):
    # Do cuda() before wrapping with variable
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)

def replace_halogen(string):
    """Regex to replace Br and Cl with single letters"""
    br = re.compile('Br')
    cl = re.compile('Cl')
    string = br.sub('R', string)
    string = cl.sub('L', string)
    return string

def letterToIndex(letter,smiles_letters):
    if letter == 'l':
        letter = 'L'
    return smiles_letters.index(letter)

def line2voc_arr(line,letters):
    arr = []
    regex = '(\[[^\[\]]{1,10}\])'
    
    line = replace_halogen(line)
    char_list = re.split(regex, line)
    for li, char in enumerate(char_list):
        if char.startswith('['):
               arr.append(letterToIndex(char,letters)) 
        else:
            chars = [unit for unit in char]
            for i, unit in enumerate(chars):
                arr.append(letterToIndex(unit,letters))
           
    return arr, len(arr)

def pad_sequences(vectorized_seqs_ligand, seq_lengths_ligand, vectorized_seqs_pocket, seq_lengths_pocket, properties, max_ligand_len, max_pocket_len):
    # ligand_seq_length = max(seq_lengths_ligand)
    # pocket_seq_length = max(seq_lengths_pocket)

    '''
     need to modify
    '''
    ligand_seq_length = max_ligand_len
    pocket_seq_length = max_pocket_len
  
    
    seq_tensor = torch.zeros((len(vectorized_seqs_ligand), ligand_seq_length+pocket_seq_length)).long()
    for idx, (seq_lig, lig_len, seq_poc, poc_len) in enumerate(zip(vectorized_seqs_ligand, seq_lengths_ligand, vectorized_seqs_pocket, seq_lengths_pocket)):
        if lig_len > ligand_seq_length:
            seq_lig = seq_lig[:ligand_seq_length]
        seq_lig = np.array(seq_lig)
        lig_np = np.zeros(ligand_seq_length)
        lig_np[:len(seq_lig)] = seq_lig

        if poc_len > pocket_seq_length:
            seq_poc = seq_poc[:pocket_seq_length]
        seq_poc = np.array(seq_poc)
        poc_np = np.zeros(pocket_seq_length)
        poc_np[:len(seq_poc)] = seq_poc
        
        
        smiles = np.concatenate((lig_np, poc_np), axis=0)
        seq_tensor[idx] = torch.LongTensor(smiles)
   
    target = properties.double()
    # if len(properties):
    #     target = target[perm_idx]
    # Return variables
    # DataParallel requires everything to be a Variable
    return create_variable(seq_tensor), create_variable(target)

def pad_sequences_(vectorized_seqs_ligand, seq_lengths_ligand, vectorized_seqs_pocket, seq_lengths_pocket, properties, max_ligand_len, max_pocket_len):
    # ligand_seq_length = max(seq_lengths_ligand)
    # pocket_seq_length = max(seq_lengths_pocket)

    '''
     need to modify
    '''
    ligand_seq_length = max_ligand_len
    pocket_seq_length = max_pocket_len
  
    
    seq_tensor = torch.zeros((len(vectorized_seqs_ligand), ligand_seq_length+pocket_seq_length)).long()
    for idx, (seq_lig, lig_len, seq_poc, poc_len) in enumerate(zip(vectorized_seqs_ligand, seq_lengths_ligand, vectorized_seqs_pocket, seq_lengths_pocket)):
        if lig_len > ligand_seq_length:
            seq_lig = seq_lig[:ligand_seq_length]
        seq_lig = np.array(seq_lig)
        lig_np = np.zeros(ligand_seq_length)
        lig_np[:len(seq_lig)] = seq_lig

        if poc_len > pocket_seq_length:
            seq_poc = seq_poc[:pocket_seq_length]
        seq_poc = np.array(seq_poc)
        poc_np = np.zeros(pocket_seq_length)
        poc_np[:len(seq_poc)] = seq_poc
        
        
        smiles = np.concatenate((lig_np, poc_np), axis=0)
        seq_tensor[idx] = torch.LongTensor(smiles)
   
    target = properties.float()
    # if len(properties):
    #     target = target[perm_idx]
    # Return variables
    # DataParallel requires everything to be a Variable
    return seq_tensor, target

def make_variables(ligand, pocket, properties, letters, max_ligand_len, max_pocket_len):
  
    properties = torch.tensor([float(item) for item in properties])
    sequence_and_length_ligand = [line2voc_arr(line,letters) for line in ligand]
    vectorized_seqs_ligand = [sl[0] for sl in sequence_and_length_ligand]
    seq_len = []
    for vector in vectorized_seqs_ligand:
        seq_len.append(len(vector))
    seq_lengths_ligand = torch.LongTensor(seq_len)


    sequence_and_length_pocket = [line2voc_arr(line,letters) for line in pocket]
    vectorized_seqs_pocket = [sl[0] for sl in sequence_and_length_pocket]
    seq_len = []
    for vector in vectorized_seqs_pocket:
        seq_len.append(len(vector))
    seq_lengths_pocket = torch.LongTensor(seq_len)

    return pad_sequences(vectorized_seqs_ligand, seq_lengths_ligand, vectorized_seqs_pocket, seq_lengths_pocket, properties, max_ligand_len, max_pocket_len)


def make_variables_(ligand, pocket, properties, letters, max_ligand_len, max_pocket_len):
  
    properties = torch.tensor([float(item) for item in properties])
    sequence_and_length_ligand = [line2voc_arr(line,letters) for line in ligand]
    vectorized_seqs_ligand = [sl[0] for sl in sequence_and_length_ligand]
    seq_len = []
    for vector in vectorized_seqs_ligand:
        seq_len.append(len(vector))
    seq_lengths_ligand = torch.LongTensor(seq_len)


    sequence_and_length_pocket = [line2voc_arr(line,letters) for line in pocket]
    vectorized_seqs_pocket = [sl[0] for sl in sequence_and_length_pocket]
    seq_len = []
    for vector in vectorized_seqs_pocket:
        seq_len.append(len(vector))
    seq_lengths_pocket = torch.LongTensor(seq_len)

    return pad_sequences_(vectorized_seqs_ligand, seq_lengths_ligand, vectorized_seqs_pocket, seq_lengths_pocket, properties, max_ligand_len, max_pocket_len)

def make_variable_llinas(smiles, properties, letters, max_smiles_len):
    properties = [float(item) for item in properties]
    sequence_and_length_smiles = [line2voc_arr(line,letters) for line in smiles]
    vectorized_seqs_smiles = [sl[0] for sl in sequence_and_length_smiles]
    
    seq_len = []
    for vector in vectorized_seqs_smiles:
        seq_len.append(len(vector))
    max_smiles_len = max(seq_len)
    seq_smiles = np.zeros((len(vectorized_seqs_smiles), max_smiles_len))
    for idx, smi in enumerate(vectorized_seqs_smiles):
        seq_smi = np.array(smi)
        seq_smiles[idx, :len(smi)] = seq_smi
    return smiles, seq_smiles, properties  
    
 
        
    


def getSeqLen(data, letters):
    ligandList = []
    pocketList = []
    for line in data:
        ligandList.append(line[0])
        pocketList.append(line[1])
    
    sequence_and_length_ligand = [line2voc_arr(line,letters) for line in ligandList]
    vectorized_seqs_ligand = [sl[0] for sl in sequence_and_length_ligand]
    seq_len_ligand = []
    for vector in vectorized_seqs_ligand:
        seq_len_ligand.append(len(vector))

    sequence_and_length_pocket = [line2voc_arr(line,letters) for line in pocketList]
    vectorized_seqs_pocket = [sl[0] for sl in sequence_and_length_pocket]
    seq_len_pocket = []
    for vector in vectorized_seqs_pocket:
        seq_len_pocket.append(len(vector))    
    return max(seq_len_ligand), max(seq_len_pocket)

def gene_ECFP(nameTuple):
    nameList = list(nameTuple)
    lig_ecfp_List = []
    poc_ecfp_List = []
    for name in nameList:
        ligand = '/home/bioinfor3/Lxh/PDBind_2020/refined-set/{}/{}_ligand.mol2'.format(name, name)
        pocket = '/home/bioinfor3/Lxh/PDBind_2020/refined-set/{}/{}_pocket.pdb'.format(name, name)

        # 读取MOL2文件并创建分子对象
        ligand_mol_supplier = Chem.MolFromMol2File(ligand)
        # 读取PDB文件并创建分子对象
        pocket_mol_supplier = Chem.MolFromPDBFile(pocket)


        ligand_ecfp = AllChem.GetMorganFingerprintAsBitVect(ligand_mol_supplier, radius=2, nBits=1024)
        pocket_ecfp = AllChem.GetMorganFingerprintAsBitVect(pocket_mol_supplier, radius=2, nBits=1024)

        lig_ecfp_List.append(np.array(ligand_ecfp))
        poc_ecfp_List.append(np.array(pocket_ecfp))
    
    # lig_combined_array = np.concatenate(lig_ecfp_List, axis=0)
    # poc_combined_array = np.concatenate(poc_ecfp_List, axis=0)

    lig_ecfp_tensor = torch.tensor(lig_ecfp_List)
    poc_ecfp_tensor = torch.tensor(poc_ecfp_List)
    
    return create_variable(lig_ecfp_tensor), create_variable(poc_ecfp_tensor)

def make_ecfp(ligand_ecfp, pocket_ecfp):
    ligand_length = 1024
    pocket_length = 1024

    ecfp_tensor = torch.zeros((len(ligand_ecfp), ligand_length+pocket_length)).float()


    # my_list = [[0] * 100 for _ in range(10)]
    for idx, (seq_lig, seq_poc) in enumerate(zip(ligand_ecfp, pocket_ecfp)):
        ecfp_lig = seq_lig.split(',')
        ecfp_lig = [int(val) for val in ecfp_lig]
        ecfp_poc = seq_poc.split(',')
        ecfp_poc = [int(val) for val in ecfp_poc]
        ecfp = ecfp_lig + ecfp_poc
        ecfp_tensor[idx] = torch.LongTensor(ecfp)

    return create_variable(ecfp_tensor)

def make_ecfp_(ligand_ecfp, pocket_ecfp):
    ligand_length = 1024
    pocket_length = 1024

    ecfp_tensor = torch.zeros((len(ligand_ecfp), ligand_length+pocket_length)).float()


    # my_list = [[0] * 100 for _ in range(10)]
    for idx, (seq_lig, seq_poc) in enumerate(zip(ligand_ecfp, pocket_ecfp)):
        ecfp_lig = seq_lig.split(',')
        ecfp_lig = [int(val) for val in ecfp_lig]
        ecfp_poc = seq_poc.split(',')
        ecfp_poc = [int(val) for val in ecfp_poc]
        ecfp = ecfp_lig + ecfp_poc
        ecfp_tensor[idx] = torch.LongTensor(ecfp)

    return ecfp_tensor

def make_ecfp_single(data):
    smi_length = 1024
    ecfp_tensor = torch.zeros((len(data), smi_length)).float()
    for i in range(len(data)):
        ecfp_ = data[i].split(',')
        ecfp_ = [int(val) for val in ecfp_]
        ecfp_tensor[i] = torch.LongTensor(ecfp_)

    return ecfp_tensor

def make_ecfp_single_(data):
    smi_length = 1024
    ecfp_array = np.zeros((len(data), smi_length), dtype=np.float32)
    for i in range(len(data)):
        ecfp_ = data[i].split(',')
        ecfp_ = [int(val) for val in ecfp_]
        ecfp_array[i] = np.array(ecfp_, dtype=np.float32)
    return ecfp_array

def transEnsmiToTorch_single(data, max_smiles_len):
    smi_tensor = torch.zeros((len(data), max_smiles_len)).int()
    for i in range(len(data)):
        enSmi = data[i].split(',')
        enSmi = [int(float(val)) for val in enSmi]
        smiArray = np.zeros(max_smiles_len)
        smiArray[:len(enSmi)] = enSmi
        smi_tensor[i] = torch.LongTensor(smiArray)
    return smi_tensor

def transEnsmiToNumpy_single(data, max_smiles_len):
    smi_array = np.zeros((len(data), max_smiles_len), dtype=np.int32)
    for i in range(len(data)):
        enSmi = data[i].split(',')
        enSmi = [int(float(val)) for val in enSmi]
        smi_array[i, :len(enSmi)] = enSmi
    return smi_array


def transPropToTorch(data):
    data_list = [float(item) for item in data]
    tensor_data = torch.Tensor(data_list)
    return tensor_data



def make_variable_one(smiles, letters, max_smiles_len):
    resultVec = []
    char_list = tokenizer(smiles)
    for item in char_list:
        resultVec.append(letters[item])
        

 
    if len(resultVec) < max_smiles_len:
        resultVec.extend([0] * (max_smiles_len - len(resultVec)))
    elif len(resultVec) > max_smiles_len:
        resultVec = resultVec[:max_smiles_len]
    return resultVec
