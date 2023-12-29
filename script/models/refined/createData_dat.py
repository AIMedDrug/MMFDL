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

    savePath = '../../../dataSour/refined/dataTrain.dat'
    with open(savePath, 'w') as dat_file:
        for row in dataTrain:
            smi_lig = row[0]
            smi_poc = row[1]
            enSmi_lig = make_variable_one(smi_lig, smilesVoc, 130)
            enSmi_poc = make_variable_one(smi_poc, smilesVoc, 1400)
            enSmi = enSmi_lig + enSmi_poc
            encodedSmi = ','.join(map(str, enSmi))
            ep_lig = row[2]
            ep_poc = row[3]
            ecfp = ep_lig + ',' +ep_poc
            logkd = row[4]
            line = smi_lig + ' ' + smi_poc + ' ' + encodedSmi + ' ' + ecfp + ' ' + logkd
            dat_file.write(line + '\n')
    print('writing ' + savePath + ' finished!')
    
    savePath = '../../../dataSour/refined/dataTest.dat'
    with open(savePath, 'w') as dat_file:
        for row in dataTest:
            smi_lig = row[0]
            smi_poc = row[1]
            enSmi_lig = make_variable_one(smi_lig, smilesVoc, 130)
            enSmi_poc = make_variable_one(smi_poc, smilesVoc, 1400)
            encodedSmi = enSmi_lig + enSmi_poc
            enSmi = ','.join(map(str, encodedSmi))
            ep_lig = row[2]
            ep_poc = row[3]
            ecfp = ep_lig + ',' +ep_poc
            logkd = row[4]
            line = smi_lig + ' ' + smi_poc + ' ' + enSmi + ' ' + ecfp + ' ' + logkd
            dat_file.write(line + '\n')  
    print('writing ' + savePath + ' finished!')    


def check_ecfp(filePath):
    smiles = pd.read_csv(filePath)['smiles'].tolist()
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
    if mol == None:
        return ''
    radius = 2  # ECFP °ë¾¶
    nBits = 1024  # Ö¸ÎÆÎ»Êý
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return ecfp
    
   

def transToDat_All(filePath, savePath):
    df = pd.read_csv(filePath)
    ligand = df['ligand'].tolist()
    pocket = df['pocket'].tolist()
    logkd = df['binding_pki'].tolist()
    with open(savePath, "w") as dat_file:
        # # Ð´Èë±êÌâÐÐ
        # dat_file.write(" ".join(head) + "\n")    
        for lig, poc, kd in zip(ligand, pocket, logkd):
            ecfp_lig = gene_ecfp(lig)
            ecfp_poc = gene_ecfp(poc)
            if ecfp_lig == '' or ecfp_poc == '':
                continue
            ep_lig = ','.join(str(x) for x in ecfp_lig)
            
            ep_poc = ','.join(str(x) for x in ecfp_poc)
            # yy = "{:05.2f}".format(float(y_))
            yy = str(kd)
            line = lig + ' ' + poc + ' ' + ep_lig + ' ' +  ep_poc + ' ' + yy
            dat_file.write(line + '\n')
    print('writing ' + savePath + ' finished!')

    

if __name__ == '__main__':
    filePath = '../../../dataSour/refined/smiles_refined_set.csv'
    savePath = '../../../dataSour/refined/dataAll.dat'
    transToDat_All(filePath, savePath)
    vocab_path = '../../../dataSour/refined/smiles_char_dict.pkl'
    with open(vocab_path, 'rb') as f:
        smilesVoc = pickle.load(f)
    filePath = '../../../dataSour/refined/dataAll.dat'
    splitData_dat(filePath, smilesVoc)


