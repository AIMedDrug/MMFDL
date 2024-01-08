# -*- coding: iso-8859-1 -*-
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool as gmp
from torch_geometric.nn import GATConv, global_add_pool as gap
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, max_length):
        super(PositionalEncoding, self).__init__()
        self.embedding_size = embedding_size
        self.max_length = max_length
        
        # Compute the positional encodings once in log space
        pe = torch.zeros(max_length, embedding_size)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-math.log(10000.0) / embedding_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x * math.sqrt(self.embedding_size)
        seq_len = x.size(1)
        pe = self.pe[:, :seq_len, :]
        x = x + pe
        return x

class modelLstm(nn.Module):
    def __init__(self, args):
        super(modelLstm, self).__init__()
        
        self.num_features = args['num_features']
        self.embed_dim = args['embed_dim']
        self.output_dim = args['output_dim']
        self.dropout = args['dropout']
        self.hidden_dim_lstm = args['hidden_dim_lstm']
        self.n_output = args['n_output']
     

        
        self.embedding_smi = nn.Embedding(self.num_features, self.embed_dim)
        self.positional_encoding = PositionalEncoding(self.embed_dim, self.num_features)
        self.smi_gru = nn.GRU(self.embed_dim, self.hidden_dim_lstm, 2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(self.num_features, self.n_output)
        self.fc2 = nn.Linear(self.hidden_dim_lstm*2, 1)
        
        # self.fc2 = nn.Linear(self.output_dim, self.n_output)
    
    def forward(self, encodedSmi):
        
        embedded_smi = self.embedding_smi(encodedSmi)
        embedded_smi = self.positional_encoding(embedded_smi)
        smi_out, _ = self.smi_gru(embedded_smi)
        smi_out = self.dropout(smi_out)
        smi_out = smi_out.reshape(smi_out.shape[0], smi_out.shape[2], -1)
        out = self.fc1(smi_out)
        out = out.view(out.shape[0], -1)
        out = self.fc2(out)
        out = out.view(1, -1)
        return out


class model_BiGRU(nn.Module):
    def __init__(self, args):
        super(model_BiGRU, self).__init__()
        
        self.num_features = args['num_features']
        self.output_dim = args['output_dim']
        self.dropout = args['dropout']
        self.hidden_dim_lstm = args['hidden_dim_lstm']
        self.n_output = args['n_output']
        self.ep_gru = nn.GRU(self.num_features, self.hidden_dim_lstm, 2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(self.output_dim, self.n_output)
        self.num_attention_heads = 2
        self.ep_attention_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(2 * self.hidden_dim_lstm, self.hidden_dim_lstm),
            nn.Tanh(),
            nn.Linear(self.hidden_dim_lstm, self.output_dim)
        ) for _ in range(self.num_attention_heads)])
        self.ep_fc_layers = nn.ModuleList([nn.Linear(self.hidden_dim_lstm * 2, self.output_dim) for _ in range(self.num_attention_heads)])
    
    def forward(self, ecfp):
        ep_out, _ = self.ep_gru(ecfp)
        ep_attended_out = 0
        for i in range(self.num_attention_heads):
            ep_attention_weights = torch.softmax(self.ep_attention_layers[i](ep_gru), dim=1)
            ep_linear = self.ep_fc_layers[i](ep_gru)
            ep_attended_out += ep_attention_weights * ep_linear
        ep_attended_out /= self.num_attention_heads
        out = self.fc(ep_out)
        out = out.view(1, -1)
        return out

class modelDnn(nn.Module):
    def __init__(self, args):
        super(modelDnn, self).__init__()
        
        self.num_features = args['num_features']
        self.output_dim = args['output_dim']
        self.dropout = args['dropout']
        self.n_output = args['n_output']
    
        
        self.fc1 = nn.Linear(self.num_features, self.output_dim)
        # 添加卷积层
        self.conv1 = nn.Conv1d(in_channels=self.num_features, out_channels=self.output_dim, kernel_size=1)
        
        self.fc2 = nn.Linear(self.output_dim, self.output_dim)  # 适应卷积输出的维度
        self.dropout = nn.Dropout(self.dropout)
        self.fc3 = nn.Linear(self.output_dim, 1)
    
    def forward(self, ecfp):
        # fc1_out = self.fc1(ecfp)
        # fc1_out = F.relu(fc1_out)  
        
 
        ecfp = ecfp.view(ecfp.size(0), -1, 1)  # 为卷积层增加一维
        ecfp = self.conv1(ecfp)
        # ecfp = F.relu(ecfp)
        # ecfp = F.max_pool1d(ecfp, kernel_size=1)  
        ecfp = ecfp.view(ecfp.shape[0], -1)

        # fc2_out = self.fc2(ecfp)
        # fc2_out = F.relu(fc2_out)
        out = self.fc3(ecfp)
        out = out.view(1, -1)
        # ep_out = self.fc1(ecfp)
        # ep_out = self.dropout(ep_out)
        # ep_out = self.fc2(ep_out)
        # out = ep_out.view(1, -1)
        return out

class modelGcn(nn.Module):
    def __init__(self, args):
        super(modelGcn, self).__init__()
        
        self.num_features = args['num_features']
        self.output_dim = args['output_dim']
        self.n_output = args['n_output']
        self.dropout = args['dropout']
     
        self.gcn_conv1 = GCNConv(self.num_features, self.num_features)
        self.gcn_conv2 = GCNConv(self.num_features, self.num_features * 2)
        self.relu = nn.ReLU()
        
        self.dropout = nn.Dropout(self.dropout)
        self.fc = nn.Linear(self.num_features* 2, self.n_output)
        
    
    def forward(self, x, edge_index, batch):
        x = self.gcn_conv1(x, edge_index)
        x = F.relu(x)
        x = self.gcn_conv2(x, edge_index)
        x = F.relu(x)
        gc = gap(x, batch)
        gc = self.dropout(gc)
        gc_out = self.fc(gc)
        out = gc_out.view(1, -1)
        return out
