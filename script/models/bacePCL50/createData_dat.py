# -*- coding: iso-8859-1 -*-
import pandas as pd
import csv
from rdkit import Chem
from rdkit.Chem import AllChem
from utils_smiecfp import *
from data_gen_modify import *

# from data_gen_modify import *
def splitData_dat(filePath, smilesVoc):
    allData = []
    # ´ò¿ª .dat ÎÄ¼þÒÔ¶ÁÈ¡Ä£Ê½
    with open(filePath, "r") as dat_file:
        # ÖðÐÐ¶ÁÈ¡ÎÄ¼þÄÚÈÝ
        for line in dat_file:
            # È¥³ý»»ÐÐ·û²¢ÒÔ¿Õ¸ñ·Ö¸î×Ö·û
            line = line.strip()  # È¥³ý»»ÐÐ·ûºÍ¿Õ°××Ö·û
            elements = line.split(" ")  # ÒÔ¿Õ¸ñ·Ö¸î×Ö·û
            allData.append(elements)
    #Ëæ»ú´òÂÒ
    random.shuffle(allData)
    count = int(len(allData) / 7)
    dataTrain = []
    dataTest = []
    for i in range(count):
        for j in range(7):
            if j < 6:
                dataTrain.append(allData[i*7+j])
            else:
                dataTest.append(allData[i*7+j])


    savePath = '../../../dataSour/bacePLC50/dataTrain.dat'
    with open(savePath, 'w') as dat_file:
        for row in dataTrain:
            smi = row[1]
            enSmi = make_variable_one(smi, smilesVoc, 130)
            enSmi = ','.join(map(str, enSmi))
            ep = row[2]
            exp = row[3]
            line = smi + ' ' + enSmi + ' ' + ep + ' ' + exp
            dat_file.write(line + '\n')
    print('writing ' + savePath + ' finished!')

    savePath = '../../../dataSour/bacePLC50/dataTest.dat'
    with open(savePath, 'w') as dat_file:
        for row in dataTest:
            smi = row[1]
            enSmi = make_variable_one(smi, smilesVoc, 130)
            enSmi = ','.join(map(str, enSmi))
            ep = row[2]
            exp = row[3]
            line = smi + ' ' + enSmi + ' ' + ep + ' ' + exp
            dat_file.write(line + '\n')   
    print('writing ' + savePath + ' finished!')    


def check_ecfp(filePath):
    smiles = pd.read_csv(filePath)['mol'].tolist()
    notGen = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol != None:
            # ¼ì²éÊÇ·ñ´æÔÚÇâÔ­×Ó
            if mol.HasSubstructMatch(Chem.MolFromSmarts("[H]")):
                mol = Chem.RemoveHs(mol)
            ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            if ecfp == None:
                notGen.append(smi)
        else:
            notGen.append(smi)

    print(notGen)


def gene_ecfp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    radius = 2  # ECFP °ë¾¶
    nBits = 1024  # Ö¸ÎÆÎ»Êý
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return ecfp
    
   

def transToDat_All(filePath, savePath):
    df = pd.read_csv(filePath)
    label = []
    smiles = []
    y = []
    for lab, smi, yy in zip(df['Model'].tolist(), df['mol'].tolist(), df['pIC50'].tolist()):
        label.append(lab)
        smiles.append(smi)
        y.append(yy)
    with open(savePath, "w") as dat_file:
        for lab, smi, y_ in zip(label, smiles, y):
            ecfp = gene_ecfp(smi)
            ep = ','.join(str(x) for x in ecfp)
            # yy = "{:05.2f}".format(float(y_))
            yy = str(y_)
            line = lab + ' ' + smi + ' ' + ep + ' ' + yy
            dat_file.write(line + '\n')
    print('writing ' + savePath + ' finished!')

def transToDat_encode(filePath, savePath, smilesVoc, max_smiles_len):
    df = pd.read_csv(filePath)
    smiles = df['smiles'].tolist()
    y = df['y'].tolist()
    head = ['smiles', 'encodedSmi', 'ecfp', 'properties']
    smiles, encodedSmi, properties  = make_variable_llinas(smiles, y, smilesVoc, max_smiles_len)
    with open(savePath, "w") as dat_file:
        # Ð´Èë±êÌâÐÐ
        # dat_file.write(" ".join(head) + "\n")    
        for smi, enSmi, prop in zip(smiles, encodedSmi, properties):
            
            ecfp = gene_ecfp(smi)
            ep = ','.join(str(x) for x in ecfp)
            enSmi = ','.join(str(x) for x in enSmi)
            prop = str(prop)
            line = smi + ' ' + enSmi + ' ' + ep + ' ' + prop
            dat_file.write(line + '\n')
    print('writing ' + savePath + ' finished!')
    


if __name__ == '__main__':
    filePath = '../../../dataSour/bacePLC50/bace.csv'
    check_ecfp(filePath)

    filePath = '../../../dataSour/bacePLC50/bace.csv'
    savePath = '../../../dataSour/bacePLC50/dataAll.dat'
    transToDat_All(filePath, savePath)

    vocab_path = '../../../dataSour/bacePLC50/smiles_char_dict.pkl'
    with open(vocab_path, 'rb') as f:
        smilesVoc = pickle.load(f)
    
    filePath = '../../../dataSour/bacePLC50/dataAll.dat'
    splitData_dat(filePath, smilesVoc)


