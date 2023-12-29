# -*- coding: iso-8859-1 -*-
# from myTrCPI.script.makeModel.utils_smiecfp import *
from sklearn import metrics
# from torch.utils.data import Dataset, DataLoader
# from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric.loader import DataLoader

from utils_smiecfp import *
from data_gen_modify import *
from analysis import *
from model_combination_trlsGc_weight import *
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
import warnings
from sklearn.linear_model import ElasticNet
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")


gpu_index = 1  # ¿ÉÒÔ¸ù¾ÝÐèÒª¸ü¸ÄÎªÆäËû GPU ±àºÅ

# ¼ì²éÊÇ·ñÓÐ¿ÉÓÃµÄ GPU Éè±¸
if torch.cuda.is_available():
    # ÉèÖÃÎªÖ¸¶¨µÄ GPU Éè±¸
    device = torch.device(f'cuda:{gpu_index}')
else:
    # Èç¹ûÃ»ÓÐ¿ÉÓÃµÄ GPU£¬Ê¹ÓÃ CPU
    device = torch.device('cpu')




epochs = 50
batch_size = 16
label = 10000
random_state= 42

argsCom = {
    'num_features_smi': 1530,
    'num_features_ecfp':2048,
    'num_features_x': 78,
    'dropout': 0.1, 
    'num_layer': 2,
    'num_heads': 2,
    'hidden_dim': 256,
    'output_dim': 128,
    'n_output': 1
    
}



resultLoss = {'losses_train': [], 'losses_val': []}



train_data = formDataset(root='../../../dataSour/refined', dataset='data_train')
train_ratio = 0.8
num_data = len(train_data)
indices = list(range(num_data))
train_indices, val_indices = train_test_split(indices, train_size=train_ratio, shuffle=True, random_state=random_state)
train_dataset = [train_data[i] for i in train_indices]
val_dataset = [train_data[i] for i in val_indices]
trainLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valLoader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)



learning_rate = 0.0001
com_model = comModel(argsCom).to(device)
optimizer_com = torch.optim.Adam(com_model.parameters(), lr=learning_rate)
criterion_com = torch.nn.MSELoss()

com_model.train()
for i in range(epochs):
    print("Running EPOCH",i+1)
    total_loss = 0
    n_batches = 0
    correct = 0
    '''
        train²¿·Ö
    '''
    for batch_idx, data in enumerate(trainLoader):
        encodedSmi = torch.LongTensor(data.smi).to(device)
        encodedSmi_mask = torch.LongTensor(getInput_mask(data.smi)).to(device)
        ecfp = torch.FloatTensor(data.ep).to(device)
        y = data.y.to(device)
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        batch = data.batch.to(device)
        y_pred = com_model(encodedSmi, encodedSmi_mask, ecfp, x, edge_index, batch)
        loss1 = criterion_com(y_pred[0].type(torch.DoubleTensor), y.type(torch.DoubleTensor))
        loss2 = criterion_com(y_pred[1].type(torch.DoubleTensor), y.type(torch.DoubleTensor))
        loss3 = criterion_com(y_pred[2].type(torch.DoubleTensor), y.type(torch.DoubleTensor))
        loss = (loss1 + loss2 + loss3) / 3
        total_loss += (loss.data)/3
        optimizer_com.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(com_model.parameters(),0.5)
        optimizer_com.step()
        n_batches+=1
    avg_loss = total_loss / n_batches
    resultLoss['losses_train'].append(avg_loss)
    print('train avg_loss is: ', avg_loss.item())

    '''
        val²¿·Ö
    '''
    total_loss = 0
    n_batches = 0
    correct = 0
    for batch_idx, data in enumerate(valLoader):
        encodedSmi = torch.LongTensor(data.smi).to(device)
        encodedSmi_mask = torch.LongTensor(getInput_mask(data.smi)).to(device)
        ecfp = torch.FloatTensor(data.ep).to(device)
        y = data.y.to(device)
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        batch = data.batch.to(device)
        y_pred = com_model(encodedSmi, encodedSmi_mask, ecfp, x, edge_index, batch) 
        loss1 = criterion_com(y_pred[0].type(torch.DoubleTensor), y.type(torch.DoubleTensor))
        loss2 = criterion_com(y_pred[1].type(torch.DoubleTensor), y.type(torch.DoubleTensor))
        loss3 = criterion_com(y_pred[2].type(torch.DoubleTensor), y.type(torch.DoubleTensor))
        loss = (loss1 + loss2 + loss3) / 3
        total_loss += (loss.data)/3
        optimizer_com.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(com_model.parameters(),0.5)
        optimizer_com.step()
        n_batches+=1
    avg_loss = total_loss / n_batches
    resultLoss['losses_val'].append(avg_loss) 
    print('val avg_loss is: ', avg_loss.item())
    print('\n')

val_data = []
pred_data1 = []
pred_data2 = []
pred_data3 = []
com_model.eval()
for batch_idx, data in enumerate(valLoader):
    encodedSmi = torch.LongTensor(data.smi).to(device)
    encodedSmi_mask = torch.LongTensor(getInput_mask(data.smi)).to(device)
    ecfp = torch.FloatTensor(data.ep).to(device)
    y = data.y.to(device)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    batch = data.batch.to(device)
    y_pred = com_model(encodedSmi, encodedSmi_mask, ecfp, x, edge_index, batch)
    val_data.append(y.tolist())
    pred_data1.append(y_pred[0].tolist())
    pred_data2.append(y_pred[1].tolist())
    pred_data3.append(y_pred[2].tolist())
'''
    Weight analysis 
'''
def flattened_data(data):
    fla_data = [item for sublist in data for item in sublist]
    merged_data = np.array(fla_data).flatten() 
    return merged_data

data_ = {}
data_['pred1'] = flattened_data(pred_data1)
data_['pred2'] = flattened_data(pred_data2)
data_['pred3'] = flattened_data(pred_data3)
data_['true'] = flattened_data(val_data)
val_true = data_['true']
val_pred = np.column_stack((data_['pred1'], data_['pred2'], data_['pred3']))

lasso_model = Lasso(alpha=0.5)
lasso_model.fit(val_pred, val_true)
lasso_weights = lasso_model.coef_
print("Lasso Weights:", lasso_weights)

elastic_net = ElasticNet(alpha=0.3, l1_ratio=0.6)
elastic_net.fit(val_pred, val_true)
elastic_weights = elastic_net.coef_
print("elastic weight", elastic_weights)

RF = RandomForestRegressor()
RF.fit(val_pred, val_true)
RF_importances = RF.feature_importances_
print("RF weight", RF_importances)

gradientboost = GradientBoostingRegressor(n_estimators=40, learning_rate=0.1, max_depth=3, random_state=random_state)
gradientboost.fit(val_pred, val_true)
gradientboost_weights = gradientboost.feature_importances_
print("gradientboost_weights:", gradientboost_weights)



test_data = formDataset(root='../../../dataSour/refined', dataset='data_test')
testLoader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)

sour_data = []
pred_data1 = []
pred_data2 = []
pred_data3 = []
com_model.eval()
for batch_idx, data in enumerate(testLoader):
    encodedSmi = torch.LongTensor(data.smi).to(device)
    encodedSmi_mask = torch.LongTensor(getInput_mask(data.smi)).to(device)
    ecfp = torch.FloatTensor(data.ep).to(device)
    y = data.y.to(device)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    batch = data.batch.to(device)
    y_pred = com_model(encodedSmi, encodedSmi_mask, ecfp, x, edge_index, batch)  
    sour_data.append(y.tolist())
    pred_data1.append(y_pred[0].tolist())
    pred_data2.append(y_pred[1].tolist())
    pred_data3.append(y_pred[2].tolist()) 





y_lasso = lasso_weights[0] * flattened_data(pred_data1) + lasso_weights[1] * flattened_data(pred_data2) + lasso_weights[2] * flattened_data(pred_data3)
y_ela = elastic_weights[0] * flattened_data(pred_data1) + elastic_weights[1] * flattened_data(pred_data2) + elastic_weights[2] * flattened_data(pred_data3)
y_tree = RF_importances[0] * flattened_data(pred_data1) + RF_importances[1] * flattened_data(pred_data2) + RF_importances[2] * flattened_data(pred_data3)
y_gradient = gradientboost_weights[0] * flattened_data(pred_data1) + gradientboost_weights[1] * flattened_data(pred_data2) + gradientboost_weights[2] * flattened_data(pred_data3)
y_sour = flattened_data(sour_data)


def calRmseMae(y, y_pred):
    yResult = {}
    ground_truth = y
    predictions = y_pred
    # calculate rmse
    squared_errors = (ground_truth - predictions) ** 2
    rmse = np.sqrt(np.mean(squared_errors))
    yResult['rmse'] = rmse
    # calculate mae
    absolute_errors = np.abs(ground_truth - predictions)
    mae = np.mean(absolute_errors)
    yResult['mae'] = mae
    return yResult

yResult = {}
yResult['lasso'] = calRmseMae(y_sour, y_lasso)
yResult['leastic'] = calRmseMae(y_sour, y_ela)
yResult['rf'] = calRmseMae(y_sour, y_tree)
yResult['gradientBoost'] = calRmseMae(y_sour, y_gradient)
print(yResult)

weightDic = {}
weightDic['lasso'] = lasso_weights
weightDic['elastic'] = elastic_weights
weightDic['rf'] = RF_importances
weightDic['gradientBoost'] = gradientboost_weights




savePath = '../../../result/seed42/refined/weight/{}_rmseMae_{}_{}_{}_weight_com.csv'.format(label, batch_size, epochs, random_state)
rmseAndMae(yResult, savePath)
savePath = '../../../result/seed42/refined/weight/{}_weight_{}_{}_{}_weight_com.csv'.format(label, batch_size, epochs, random_state)
df_weight = pd.DataFrame(weightDic)
df_weight.to_csv(savePath, index=False)


savePath = '../../../result/seed42/refined/weight/{}_scatter_predited_{}_{}_{}_lasso_com.png'.format(label, batch_size, epochs, random_state)
pltPredict_linregress(y_lasso, y_sour, savePath)
savePath = '../../../result/seed42/refined/weight/{}_scatter_predited_{}_{}_{}_elastic_com.png'.format(label, batch_size, epochs, random_state)
pltPredict_linregress(y_ela, y_sour, savePath)
savePath = '../../../result/seed42/refined/weight/{}_scatter_predited_{}_{}_{}_rf_com.png'.format(label, batch_size, epochs, random_state)
pltPredict_linregress(y_tree, y_sour, savePath)
savePath = '../../../result/seed42/refined/weight/{}_scatter_predited_{}_{}_{}_gradient_com.png'.format(label, batch_size, epochs, random_state)
pltPredict_linregress(y_gradient, y_sour, savePath)
print('\n')

result = {}
result['y_pred'] = y_lasso
result['y'] = y_sour
resultPath = '../../../result/seed42/refined/weight/{}-result-{}-{}-{}-lasso-com.csv'.format(label, batch_size, epochs, random_state)
resultDf = pd.DataFrame(result)
resultDf.to_csv(resultPath, index=False)
print(resultPath + '\tsaved successfully')
result = {}
result['y_pred'] = y_ela
result['y'] = y_sour
resultPath = '../../../result/seed42/refined/weight/{}-result-{}-{}-{}-elastic-com.csv'.format(label, batch_size, epochs, random_state)
resultDf = pd.DataFrame(result)
resultDf.to_csv(resultPath, index=False)
print(resultPath + '\tsaved successfully')
result = {}
result['y_pred'] = y_tree
result['y'] = y_sour
resultPath = '../../../result/seed42/refined/weight/{}-result-{}-{}-{}-rf-com.csv'.format(label, batch_size, epochs, random_state)
resultDf = pd.DataFrame(result)
resultDf.to_csv(resultPath, index=False)
print(resultPath + '\tsaved successfully')
result = {}
result['y_pred'] = y_gradient
result['y'] = y_sour
resultPath = '../../../result/seed42/refined/weight/{}-result-{}-{}-{}-gradient-com.csv'.format(label, batch_size, epochs, random_state)
resultDf = pd.DataFrame(result)
resultDf.to_csv(resultPath, index=False)
print(resultPath + '\tsaved successfully')
print('\n')

savePath = '../../../result/seed42/refined/weight/{}_loss_{}_{}_{}_weight_com.png'.format(label, batch_size, epochs, random_state)
plotLoss(resultLoss['losses_train'], resultLoss['losses_val'], savePath)

savePath = '../../../result/seed42/refined/weight/{}_loss_{}_{}_{}_weight_com.csv'.format(label, batch_size, epochs, random_state)
lossDf = pd.DataFrame(resultLoss)
lossDf.to_csv(savePath, index=False)


savePath = '../../../result/seed42/lipophilicity/weight/{}_validation_{}_{}_{}_data.csv'.format(label, batch_size, epochs, random_state)
data_as_lists = {key: data_[key].tolist() for key in data_}
df_data = pd.DataFrame(data_as_lists)
df_data.to_csv(savePath, index=False)
print(savePath + '\tsave succeed!')



