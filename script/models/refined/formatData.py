# -*- coding: iso-8859-1 -*-
import pandas as pd

df = pd.read_csv('../../../dataSour/refined/smiles_refined_set.csv')
resultDic = {}
resultDic['ligand'] = df['ligand']
resultDic['pocket'] = df['pocket']
resultDic['logkd'] = df['binding_pki']
res = pd.DataFrame(resultDic)
res.to_csv('../../../dataSour/refined/refined_all.csv', index=False)

smilesAll = {'smiles': []}
for item in resultDic['ligand']:
    smilesAll['smiles'].append(item)
for item in resultDic['pocket']:
    smilesAll['smiles'].append(item)


res= pd.DataFrame(smilesAll)
res.to_csv('../../../dataSour/refined/dataSmileAll.txt', index=False, header=False)
