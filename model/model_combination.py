# -*- coding: iso-8859-1 -*-
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool as gmp
from torch_geometric.nn import GATConv, global_add_pool as gap
from torch_geometric.utils import to_dense_batch
import math
import torch.nn.init as init
from model_transformer import *




# torch.cuda.set_device(0)  # ÉèÖÃÄ¬ÈÏÉè±¸ÎªµÚ¶þ¸ö GPU

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

class comModel(nn.Module):
    def __init__(self, args):
        super(comModel, self).__init__()
        self.num_features_smi = args['num_features_smi']
        self.num_features_ecfp = args['num_features_ecfp']
        self.num_features_x = args['num_features_x']
        self.dropout = args['dropout']
        self.num_layer = args['num_layer']
        self.num_heads = args['num_heads']
        self.hidden_dim = args['hidden_dim']
        self.output_dim = args['output_dim']
        self.n_output = args['n_output']
        
        self.encoder = make_model(self.num_features_smi, self.num_layer, self.hidden_dim, self.output_dim, self.num_heads, self.dropout)
        self.ep_gru = nn.GRU(self.num_features_ecfp, self.hidden_dim, self.num_layer, batch_first=True, bidirectional=True)
       
        self.smi_norm = nn.LayerNorm(self.output_dim)
        self.ep_norm1 = nn.LayerNorm(self.hidden_dim*2)
        self.ep_norm2 = nn.LayerNorm(self.output_dim)

        self.gcn_conv1 = GCNConv(self.num_features_x, self.num_features_x)
        self.gcn_conv2 = GCNConv(self.num_features_x, self.num_features_x * 2)
 
        self.num_attention_heads = 2
        self.ep_attention_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.output_dim)
        ) for _ in range(self.num_attention_heads)])
        self.ep_fc_layers = nn.ModuleList([nn.Linear(self.hidden_dim * 2, self.output_dim) for _ in range(self.num_attention_heads)])

        self.dropout = nn.Dropout()

        self.smi_fc1 = nn.Linear(self.num_features_smi, self.n_output)
        self.smi_fc2 = nn.Linear(self.hidden_dim, self.n_output)
        self.ep_fc1 = nn.Linear(self.output_dim, self.n_output)
        self.smi_ep_fc1 = nn.Linear(self.output_dim, self.num_features_x*2)
        self.gc_fc = nn.Linear(2*self.num_features_x, self.n_output)
    
    def forward(self, encodedSmi, encodedSmi_mask, ecfp, x, edge_index, batch):
        smi_mask = encodedSmi_mask.unsqueeze(1)
        smi_encoded = self.encoder(encodedSmi, smi_mask)
        smi_encoded = smi_encoded.view(smi_encoded.shape[0], smi_encoded.shape[2], -1)
        smi_ln = self.smi_fc1(smi_encoded)
        smi_ln = smi_ln.view(smi_ln.shape[0], -1)
        smi_out = self.smi_fc2(smi_ln).squeeze()
    
        ep_gru, _ = self.ep_gru(ecfp)
        ep_attended_out = 0
        for i in range(self.num_attention_heads):
            ep_attention_weights = torch.softmax(self.ep_attention_layers[i](ep_gru), dim=1)
            ep_linear = self.ep_fc_layers[i](ep_gru)
            ep_attended_out += ep_attention_weights * ep_linear
        ep_attended_out /= self.num_attention_heads
        ep_out = self.ep_fc1(ep_attended_out).squeeze()

        x = self.gcn_conv1(x, edge_index)
        x = F.relu(x)
        x = self.gcn_conv2(x, edge_index)
        x = F.relu(x)
        gc = gap(x, batch)
        gc_out = self.gc_fc(gc).squeeze()
        
        return smi_out, ep_out, gc_out



