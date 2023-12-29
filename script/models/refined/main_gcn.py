# -*- coding: iso-8859-1 -*-
# from myTrCPI.script.makeModel.utils_smiecfp import *
from sklearn import metrics
# from torch.utils.data import Dataset, DataLoader
# from torch_geometric.data import InMemoryDataset, DataLoader
from torch.utils.data import Dataset, DataLoader

from utils_smiecfp import *
from data_gen_modify import *
from analysis import *
from model_dnn_lstm_gcn import *
from utils import *
from sklearn.model_selection import train_test_split
import warnings
import pickle


gpu_index = 0  # ¿ÉÒÔ¸ù¾ÝÐèÒª¸ü¸ÄÎªÆäËû GPU ±àºÅ

# ¼ì²éÊÇ·ñÓÐ¿ÉÓÃµÄ GPU Éè±¸
if torch.cuda.is_available():
    # ÉèÖÃÎªÖ¸¶¨µÄ GPU Éè±¸
    device = torch.device(f'cuda:{gpu_index}')
else:
    # Èç¹ûÃ»ÓÐ¿ÉÓÃµÄ GPU£¬Ê¹ÓÃ CPU
    device = torch.device('cpu')


epochs = 50
batch_size = 16
label = 1000
random_state=42

argsGcn = {
    'n_output': 1,
    'num_features': 78,
    'output_dim': 128,
    'dropout': 0.1, 
}



resultLoss = {'losses_train': [], 'losses_val': []}
resultAcc = {'accs_train': [], 'accs_val': []}

argsGcn['num_features'] = 78



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
model = modelGcn(argsGcn).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()

model.train()
for i in range(epochs):
    print("Running EPOCH",i+1)
    total_loss = 0
    n_batches = 0
    correct = 0
    '''
        train²¿·Ö
    '''
    for batch_idx, data in enumerate(trainLoader):
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        batch = data.batch.to(device)
        y =data.y.to(device)
        y_pred = model(x, edge_index, batch)
        correct += torch.eq(y_pred.type(torch.DoubleTensor).squeeze(), y.type(torch.DoubleTensor)).data.sum()
        loss = criterion(y_pred.type(torch.DoubleTensor).squeeze(), y.type(torch.DoubleTensor))
        total_loss += loss.data
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),0.5)
        optimizer.step()
        n_batches+=1
    avg_loss = total_loss / n_batches
    correct = correct.type(torch.DoubleTensor)
    acc = correct.numpy()/(len(trainLoader.dataset))
    resultLoss['losses_train'].append(avg_loss)
    resultAcc['accs_train'].append(acc)
    print('train avg_loss is: ', avg_loss.item())
    print('train acc is: ', acc)

    '''
        val²¿·Ö
    '''
    total_loss = 0
    n_batches = 0
    correct = 0
    for batch_idx, data in enumerate(valLoader):
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        batch = data.batch.to(device)
        y =data.y.to(device)
        y_pred = model(x, edge_index, batch)   
        correct += torch.eq(torch.round(y_pred.type(torch.DoubleTensor).squeeze()), y.type(torch.DoubleTensor)).data.sum()
        loss = criterion(y_pred.type(torch.DoubleTensor).squeeze(), y.type(torch.DoubleTensor))
        total_loss += loss.data
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),0.5)
        optimizer.step()
        n_batches+=1
    avg_loss = total_loss / n_batches
    correct = correct.type(torch.DoubleTensor)
    acc = correct.numpy()/(len(trainLoader.dataset))
    resultLoss['losses_val'].append(avg_loss) 
    resultAcc['accs_val'].append(acc)
    print('val avg_loss is: ', avg_loss.item())
    print('val acc is: ', acc)
    print('\n')



y_pred_list = []
y_sour_list = []
correct = 0

test_data = formDataset(root='../../../dataSour/refined', dataset='data_test')
testLoader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


model.eval()
for batch_idx, data in enumerate(testLoader):
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    batch = data.batch.to(device)
    y =data.y.to(device)
    y_pred = model(x, edge_index, batch)  
    y_pred_list.append(y_pred.squeeze().tolist())
    y_sour_list.append(y.tolist())

y_pred = []
for data in y_pred_list:
    for item in data:
        value = round(item, 2)
        y_pred.append(value)

y = []
for data in y_sour_list:
    for item in data:
        value = round(item, 2)
        y.append(value)

yResult = {}
ground_truth = np.array(y)
predictions = np.array(y_pred)
# calculate rmse
squared_errors = (ground_truth - predictions) ** 2
rmse = np.sqrt(np.mean(squared_errors))
yResult['rmse'] = rmse
# calculate mae
absolute_errors = np.abs(ground_truth - predictions)
mae = np.mean(absolute_errors)
yResult['mae'] = mae
print(yResult)


savePath = '../../../result/seed42/refined/{}_rmseMae_{}_{}_{}_gcn.csv'.format(label, batch_size, epochs, random_state)
rmseAndMae(yResult, savePath)


savePath = '../../../result/seed42/refined/{}_scatter_predited_{}_{}_{}_gcn.png'.format(label, batch_size, epochs, random_state)
pltPredict_linregress(y_pred, y, savePath)
print('\n')

result = {}
result['y_pred'] = y_pred
result['y'] = y
resultPath = '../../../result/seed42/refined/{}-result-{}-{}-{}-gcn.csv'.format(label, batch_size, epochs, random_state)
resultDf = pd.DataFrame(result)
resultDf.to_csv(resultPath, index=False)
print(resultPath + '\tsaved successfully')

savePath = '../../../result/seed42/refined/{}_loss_{}_{}_{}_gcn.png'.format(label, batch_size, epochs, random_state)
plotLoss(resultLoss['losses_train'], resultLoss['losses_val'], savePath)

savePath = '../../../result/seed42/refined/{}_loss_{}_{}_{}_gcn.csv'.format(label, batch_size, epochs, random_state)
lossDf = pd.DataFrame(resultLoss)
lossDf.to_csv(savePath, index=False)







