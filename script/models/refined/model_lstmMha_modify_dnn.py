# -*- coding: iso-8859-1 -*-
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp
from torch_geometric.nn import GATConv, global_add_pool as gap
import math
import torch.nn.init as init



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

class AttentionGRU(nn.Module):
    def __init__(self, args):
        super(AttentionGRU, self).__init__()
        self.n_output = args['n_output']

        self.num_features_smi = args['num_features_smi']
        self.num_features_ecfp = args['num_features_ecfp']

        self.input_dim = args['input_dim']
        self.embed_dim = args['embed_dim']
        self.output_dim = args['output_dim']
        self.dropout = args['dropout']
        self.num_layers = args['num_layers']
        self.hidden_dim_lstm = args['hidden_dim_lstm']
        self.hidden_dim_multi = args['hidden_dim_multi']
        self.num_features_x = args['num_features_x']
        
        self.embedding_smi = nn.Embedding(self.num_features_smi, self.embed_dim)
        self.positional_encoding = PositionalEncoding(self.embed_dim, self.num_features_smi)
        
        # SMILES´¦Àí²ãºÍECFP´¦Àí²ã
        self.smi_gru = nn.GRU(self.embed_dim, self.hidden_dim_lstm, self.num_layers, batch_first=True, bidirectional=True)
        self.ep_gru = nn.GRU(self.num_features_ecfp, self.hidden_dim_lstm, self.num_layers, batch_first=True, bidirectional=True)
        self.conv1 = nn.Conv1d(in_channels=self.num_features_ecfp, out_channels=self.hidden_dim_lstm*2, kernel_size=1)

        self.smi_gru_ln = nn.LayerNorm(2 * self.hidden_dim_lstm)
        self.ep_gru_ln = nn.LayerNorm(2 * self.hidden_dim_lstm)
        self.con_ln = nn.LayerNorm(2*(self.output_dim + self.num_features_x))
        

        # Í¼´¦Àí²ã
        self.gcn_conv1 = GCNConv(self.num_features_x, self.num_features_x)
        self.gcn_conv2 = GCNConv(self.num_features_x, self.num_features_x * 2)

        self.gat_conv1 = GATConv(self.num_features_x, self.num_features_x, heads=2)
        self.gat_conv2 = GATConv(self.num_features_x*2, self.num_features_x, heads=2)
      
        
        # ¶¨Òå×¢ÒâÁ¦Í·Êý
        self.num_attention_heads = 2

        # ÎªÃ¿¸öÍ·¶¨Òå×¢ÒâÁ¦²ãºÍÏßÐÔ²ã
        self.smi_attention_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(2 * self.hidden_dim_lstm, self.hidden_dim_lstm),
            nn.Tanh(),
            nn.Linear(self.hidden_dim_lstm, self.output_dim)
        ) for _ in range(self.num_attention_heads)])

        self.ep_attention_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(2 * self.hidden_dim_lstm, self.hidden_dim_lstm),
            nn.Tanh(),
            nn.Linear(self.hidden_dim_lstm, self.output_dim)
        ) for _ in range(self.num_attention_heads)])

        self.smi_fc_layers = nn.ModuleList([nn.Linear(2 * self.hidden_dim_lstm, self.output_dim) for _ in range(self.num_attention_heads)])
        self.ep_fc_layers = nn.ModuleList([nn.Linear(2 * self.hidden_dim_lstm, self.output_dim) for _ in range(self.num_attention_heads)])

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()



        self.smi_gru_fc = nn.Linear(self.hidden_dim_lstm * 2, self.embed_dim)
        # self.ep_fc1 = nn.Linear(self.num_features_ecfp, self.embed_dim)
        self.smi_fc1 = nn.Linear(self.hidden_dim_lstm * 2, self.output_dim)
        self.smi_fc2 = nn.Linear(self.num_features_smi, self.n_output)
        self.ep_fc2 = nn.Linear(self.hidden_dim_lstm * 2, self.output_dim)
        self.out_fc1 = nn.Linear(2*(self.output_dim + self.num_features_x), self.n_output)
        init.normal_(self.out_fc1.weight, mean=0, std=0.01)
        # self.fc2 = nn.Linear(self.output_dim, self.n_output)
    
    def forward(self, encodedSmi, ecfp, x, edge_index, batch):
        embedded_smi = self.embedding_smi(encodedSmi)
        embedded_smi = self.positional_encoding(embedded_smi)
        smi_out, _ = self.smi_gru(embedded_smi)
        # smi_out = self.smi_gru_fc(smi_out)
        smi_out = self.smi_gru_ln(smi_out)


        ecfp = ecfp.view(ecfp.shape[0], -1, ecfp.shape[1])
        ep_out, _ = self.ep_gru(ecfp)
        ep_out = self.ep_gru_ln(ep_out)

        # # ep = ecfp.view(ecfp.shape[0], 1, -1)
        # # ep = self.ep_fc1(ep)
        # ep_ = ecfp.view(ecfp.shape[0], ecfp.shape[1], -1)
        # ep_out = self.conv1(ep_)
        # ep_out = ep_out.view(ep_out.shape[0], 1, -1)
        # ep_out = self.ep_gru_ln(ep_out)


        # ¶àÍ·×¢ÒâÁ¦
        smi_attended_out = 0
        ep_attended_out = 0
        for i in range(self.num_attention_heads):
            smi_attention_weights = torch.softmax(self.smi_attention_layers[i](smi_out), dim=1)
            ecfp_attention_weights = torch.softmax(self.ep_attention_layers[i](ep_out), dim=1)
            smi_linear = self.smi_fc_layers[i](smi_out)
            ep_linear = self.ep_fc_layers[i](ep_out)
            smi_attended_out += smi_attention_weights * smi_linear
            ep_attended_out += ecfp_attention_weights * ep_linear

        smi_attended_out /= self.num_attention_heads
        ep_attended_out /= self.num_attention_heads
       

        
        smi_out_fc = self.smi_fc1(smi_out)
        smi_attended_out += smi_out_fc
        ep_out_fc = self.ep_fc2(ep_out)
        ep_attended_out += ep_out_fc
        
        
        smi_attended_out = smi_attended_out.view(smi_attended_out.shape[0], smi_attended_out.shape[2], -1)
        smi_attended_out = self.smi_fc2(smi_attended_out)
        smi_attended_out = smi_attended_out.view(smi_attended_out.shape[0], -1)
        ep_attended_out = ep_attended_out.view(ep_attended_out.shape[0], -1)

        x = self.gat_conv1(x, edge_index)
        x = F.relu(x)
        x = self.gat_conv2(x, edge_index)
        x = F.relu(x)
        ga = gap(x, batch)
        
        combined_features = torch.cat((smi_attended_out, ep_attended_out, ga), dim=1)
        # combined_features = self.con_ln(combined_features)
        out = self.out_fc1(combined_features)
        out = out.view(1, -1)
        return out

# class mygat_lstmMha(torch.nn.Module):

#     def __init__(self, args):
#         super(mygat_lstmMha,self).__init__()
#          # 1D convolution on protein sequence
#         self.n_output = args['n_output']

#         self.num_features_smi = args['num_features_smi']
#         self.num_features_ecfp = args['num_features_ecfp']

#         self.embed_dim = args['embed_dim']
#         self.output_dim = args['output_dim']
#         self.dropout = args['dropout']
#         self.num_heads = args['num_heads']
#         self.hidden_dim_lstm = args['hidden_dim_lstm']
#         self.hidden_dim_multi = args['hidden_dim_multi']
#         self.num_features_x = args['num_features_x']
    

#         # self.embedding_xt = nn.Embedding(self.num_features_xt, self.embed_dim)
#         self.embedding_smi = nn.Embedding(self.num_features_smi, self.embed_dim)
#         self.positional_encoding = PositionalEncoding(self.embed_dim, self.num_features_smi)
#         self.lstm_smi = nn.LSTM(self.embed_dim, hidden_size=self.hidden_dim_lstm, num_layers=2, dropout=self.dropout, batch_first=True)
#         self.lstm_ecfp = nn.LSTM(self.num_features_ecfp, hidden_size=self.hidden_dim_lstm, num_layers=2, dropout=self.dropout, batch_first=True)
#         self.lstm_gcn = nn.LSTM(1024, hidden_size=self.hidden_dim_lstm, num_layers=2, dropout=self.dropout, batch_first=True)
#         self.attention = MultiHeadAttention(self.hidden_dim_lstm, self.hidden_dim_multi, self.num_heads)
#         self.fc_xt = nn.Linear(self.output_dim * 2, self.output_dim)
#         self.fc_out1 = nn.Linear(self.output_dim * 2, self.output_dim)
#         self.fc_out2 = nn.Linear(self.output_dim * 2, self.n_output)
#         self.dropout = nn.Dropout(self.dropout) 

#          # SMILES graph branch
#         # self.n_output = n_output
#         # self.conv1 = GCNConv(self.num_features_x, self.num_features_x)
#         # self.conv2 = GCNConv(self.num_features_x, 512)
#         # self.conv3 = GCNConv(self.num_features_x*2, self.num_features_x * 4)
#         self.conv1 = GATConv(self.num_features_x, 512, heads=2, dropout=0.1)
#         self.fc_g1 = torch.nn.Linear(self.output_dim * 2, self.output_dim)
        
        
        
#         # self.fc_g2 = torch.nn.Linear(self.hidden_dim_lstm, self.output_dim)

#         # activation and regularization
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout()     

    
#     def forward(self, encodedSmi, ecfp, x, edge_index, batch):
        
#         # process ligand
        
#         embedded_smi = self.embedding_smi(encodedSmi)
#         embedded_smi = self.positional_encoding(embedded_smi)
#         ls, _ = self.lstm_smi(embedded_smi)
#         ls = self.dropout(ls)
#         atten_ls = self.attention(ls)
#         aggregated_ls = atten_ls.mean(dim=1)
#         smi_out = self.fc_xt(aggregated_ls)
        
#         ecfp = ecfp.view(ecfp.shape[0], -1, ecfp.shape[1])
#         ep, _ = self.lstm_ecfp(ecfp)
#         ep = self.dropout(ep)
#         atten_ecfp = self.attention(ep)
#         aggregated_ep = atten_ecfp.mean(dim=1)
#         ep_out  = self.fc_xt(aggregated_ep)
      
   
#         gc = self.conv1(x, edge_index)
#         gc = self.relu(gc)
#         gc = gmp(gc, batch)       # global max pooling
#         gc = gc.view(gc.shape[0], -1, gc.shape[1])
#         gc, _ = self.lstm_gcn(gc)
#         gc = self.dropout(gc)
#         atten_gcn = self.attention(gc)
#         aggregated_gcn = atten_gcn.mean(dim=1)
#         gcn_out = self.fc_g1(aggregated_gcn)
        
#         # x = self.dropout(x)
#         # x = self.fc_g2(x)
        
#         # gcn_out = self.dropout(x)
#         con1 = torch.cat((smi_out, ep_out), dim=1)
#         out1 = self.fc_out1(con1)
#         out1 = self.dropout(out1)
#         con2 = torch.cat((out1, gcn_out), dim=1)
#         out = self.fc_out2(con2)
#         out = out.view(1, -1)
#         return out

        




# model = myDrugVQA(args)
# model = model.cuda()
# # ´´½¨²âÊÔÊý¾Ý
# smi_length = 50
# x1 = torch.randint(0, args['n_chars_seq'], (args['batch_size'], smi_length)).cuda()  # smi_length ÊÇÄãµÄ½á¹¹ÐòÁÐ³¤¶È
# x2 = torch.randint(1, args['n_chars_smi'], (args['batch_size'], smi_length)).cuda()  # smi_length ÊÇÄãµÄ½á¹¹ÐòÁÐ³¤¶È

# output = model(x1, x2)



