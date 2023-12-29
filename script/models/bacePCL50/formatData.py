# -*- coding: iso-8859-1 -*-
import pandas as pd

df = pd.read_csv('../../../dataSour/bacePLC50/bace.csv')
resultDic = {}
resultDic['mol'] = df['mol']
resultDic['plc50'] = df['pIC50']

res = pd.DataFrame(resultDic)
res.to_csv('../../../dataSour/bacePLC50/bace_all.csv', index=False)

smilesAll = {'smiles': []}
for item in resultDic['mol']:
    smilesAll['smiles'].append(item)

res= pd.DataFrame(smilesAll)
res.to_csv('../../../dataSour/bacePLC50/dataSmileAll.txt', index=False, header=False)
