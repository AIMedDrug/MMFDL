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
    'num_features_smi': 130,
    'num_features_ecfp':1024,
    'num_features_x': 78,
    'dropout': 0.1, 
    'num_layer': 2,
    'num_heads': 2,
    'hidden_dim': 256,
    'output_dim': 128,
    'n_output': 1
    
}



resultData = {'r': [], 'rmse': [], 'mae': [], 'dyna_w1': [], 'dyna_w2': [], 'dyna_w3': [], 'seed': []}
seedList = np.random.randint(0, 1001, 15)

for number in seedList:
    print('seed:  ' + str(number))
    random_state = number
    resultData['seed'].append(number)

    train_data = formDataset(root='../../../dataSour/bacePLC50', dataset='data_train')
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
            torch.cuda.empty_cache()
        avg_loss = total_loss / n_batches
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

    def flattened_data(data):
        fla_data = [item for sublist in data for item in sublist]
        merged_data = np.array(fla_data).flatten() 
        return merged_data

    # ÎªÁËÐ´ÈëvalÔ¤²âÖµºÍÕæÊµÖµ
    data_ = {}
    data_['pred1'] = flattened_data(pred_data1)
    data_['pred2'] = flattened_data(pred_data2)
    data_['pred3'] = flattened_data(pred_data3)
    data_['true'] = flattened_data(val_data)

    # ×öÈ¨ÖØ·ÖÎö
    val_data = flattened_data(val_data).astype(np.float32)
    pred_data1 = flattened_data(pred_data1).astype(np.float32)
    pred_data2 = flattened_data(pred_data2).astype(np.float32)
    pred_data3 = flattened_data(pred_data3).astype(np.float32)

    learning_rate_weight = 0.01
    weights = torch.tensor([0.2, 0.7, 0.1], requires_grad=True, dtype=torch.float)
    optimizer_weight = torch.optim.SGD([weights], lr=learning_rate_weight)
    criterion_weight = nn.MSELoss()

    val_data_tensor = torch.from_numpy(val_data)
    pred_data1_tensor = torch.from_numpy(pred_data1)
    pred_data2_tensor = torch.from_numpy(pred_data2)
    pred_data3_tensor = torch.from_numpy(pred_data2)
    print('\n')
    print('weight loss:\n')
    for i in range(epochs):
        print("Running EPOCH", i+1)
        # Compute weighted_output using torch operations, not .detach().numpy()
        if weights.data.sum() > 1:
        # Èç¹û×ÜºÍ´óÓÚ1£¬½øÐÐ¹éÒ»»¯´¦Àí
            weights.data /= weights.data.sum()
        weighted_output = (weights[0] * pred_data1_tensor + weights[1] * pred_data2_tensor + weights[2] * pred_data3_tensor).to(device)
        val_output = val_data_tensor.to(device)
        loss = criterion_weight(weighted_output, val_output)
        optimizer_weight.zero_grad()
        loss.backward()
        optimizer_weight.step()
        print('weight loss is: ', loss.item())
    print('\n')


    # ÎªÁËÐ´ÈëÈ¨ÖØ
    numpy_weights = weights.detach().numpy()
    resultData['dyna_w1'].append(numpy_weights[0])
    resultData['dyna_w2'].append(numpy_weights[1])
    resultData['dyna_w3'].append(numpy_weights[2])
    print("dynametics weight:", numpy_weights)
    
    sour_data = []
    pred_data1 = []
    pred_data2 = []
    pred_data3 = []
    test_data = formDataset(root='../../../dataSour/bacePLC50', dataset='data_test')
    testLoader1 = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)
    com_model.eval()
    for batch_idx, data in enumerate(testLoader1):
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
    yPred = numpy_weights[0] * flattened_data(pred_data1) + numpy_weights[1] * flattened_data(pred_data2) + numpy_weights[2] * flattened_data(pred_data3)
    ySour = flattened_data(sour_data)

    result = {}
    result['y_pred'] = yPred
    result['y'] = ySour
    resultPath = '../../../result/seedRandom/bacePLC50/turePred/fusion/result-SGD-{}.csv'.format(number)
    resultDf = pd.DataFrame(result)
    resultDf.to_csv(resultPath, index=False)

    
    def calRmseMae(y, y_pred):
        ground_truth = y
        predictions = y_pred
        squared_errors = (ground_truth - predictions) ** 2
        rmse = np.sqrt(np.mean(squared_errors))
        absolute_errors = np.abs(ground_truth - predictions)
        mae = np.mean(absolute_errors)
        return rmse, mae
    
    rmse, mae = calRmseMae(ySour, yPred)
    rValue = calR(yPred, ySour)
    resultData['rmse'].append(rmse)
    resultData['mae'].append(mae)
    resultData['r'].append(rValue)

    df = pd.DataFrame(resultData)
    df.to_csv('../../../result/seedRandom/bacePLC50/rmseMaeR-comDynametics.csv', index=False)






