# -*- coding: iso-8859-1 -*-
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp
from torch_geometric.nn import GATConv, global_add_pool as gap
import math
from model_transformer import *

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

class modelTransformer_smi(nn.Module):
    def __init__(self, args):
        super(modelTransformer_smi, self).__init__()
        self.num_features = args['num_features']
        self.max_features = args['max_features']
        self.dropout = args['dropout']
        self.num_layer = args['num_layer']
        self.num_heads = args['num_heads']
        self.hidden_dim = args['hidden_dim']
        self.output_dim = args['output_dim']
        self.n_output = args['n_output']

        self.encoder = make_model(self.num_features, self.num_layer, self.hidden_dim, self.output_dim, self.num_heads, self.dropout)
        self.fc1 = nn.Linear(self.num_features, self.n_output) 
        self.fc2 = nn.Linear(self.hidden_dim, self.n_output)

    def forward(self, encodedSmi, encodedSmi_mask):
        smi_mask = encodedSmi_mask.unsqueeze(1)
        smi_encoded = self.encoder(encodedSmi, smi_mask)
        smi_encoded = smi_encoded.view(smi_encoded.shape[0], smi_encoded.shape[2], -1)
        smi_ln = self.fc1(smi_encoded)
        smi_ln = smi_ln.view(smi_ln.shape[0], -1)
        out = self.fc2(smi_ln).squeeze()
        return out



class modelBigru_ep(nn.Module):
    def __init__(self, args):
        super(modelBigru_ep, self).__init__()
        
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
        ep_gru, _ = self.ep_gru(ecfp)
        ep_attended_out = 0
        for i in range(self.num_attention_heads):
            ep_attention_weights = torch.softmax(self.ep_attention_layers[i](ep_gru), dim=1)
            ep_linear = self.ep_fc_layers[i](ep_gru)
            ep_attended_out += ep_attention_weights * ep_linear
        ep_attended_out /= self.num_attention_heads
        out = self.fc(ep_attended_out)
        out = out.view(1, -1)
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
    

