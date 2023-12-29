import os
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch
from utils_smiecfp import *
from data_gen_modify import *

class formDataset(InMemoryDataset):
    def __init__(self, root='../',dataset='data_train',
                 encodedSmi=None, ecfp=None, y=None, smile_graph=None):
        super(formDataset, self).__init__(root)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(encodedSmi, ecfp, y, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, encodedSmi, ecfp, log, smile_graph):
        assert (len(encodedSmi) == len(ecfp) and len(ecfp) == len(log)), "The three lists must be the same length!"
        data_list = []
        for idx, (smi, ep, y_) in enumerate(zip(encodedSmi, ecfp, log)):
            smi = torch.LongTensor([smi])
            ep = torch.FloatTensor([ep])
            y = torch.FloatTensor([float(y_)])
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[idx]
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                )
            GCNData.smi = smi
            GCNData.ep = ep
            GCNData.y = y
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            data_list.append(GCNData)
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])