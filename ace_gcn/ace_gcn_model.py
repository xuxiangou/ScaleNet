#Import necessary modules
"""
@author: pgg
"""
import argparse
import os
import shutil
import sys
import time
import warnings
from random import sample
import pickle 
from tqdm import tqdm

import numpy as np
from sklearn import metrics

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch_scatter import scatter_mean, scatter_add
from ace_gcn.generate_dataset import GraphData, collate_pool
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
import pprint 

np.random.seed(42)
torch.manual_seed(42)

class ConvLayer(nn.Module):
    """
    Module to coduct convolutional operation on subgraphs
    """
    def __init__(self, atom_fea_len, nbr_fea_dist_len):
        """
        Define NN-classes for GCN 

        Parameters
        -------------
        atom_fea_len: int, Size of atom feature 
        nbr_fea_dist_len: int, Size of spatial bond distance encoding 
        """
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_dist_len = nbr_fea_dist_len

        self.fc_full_1 = nn.Linear(self.atom_fea_len + self.nbr_fea_dist_len, self.atom_fea_len)

        self.fc_full_2 = nn.Linear(self.atom_fea_len, self.atom_fea_len)

        self.sum_pooling = SumPooling()
        self.mean_pooling = MeanPooling()

        self.softplus1 = nn.Softplus()
        self.softplus2 = nn.Softplus()

        self.bn1 = nn.BatchNorm1d(self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)

    def forward(self, atom_fea, nbr_dist_fea, nbr_adj_value, nbr_bond_type, self_fea_idx, nbr_fea_idx, ads_atom_idx):
        """
        Forward pass for the module 

        Parameters
        ----------

        atom_fea: Chemical element identifiers - 
        nbr_dist_fea: Spatial bond feature - 
        nbr_adj_value: Value in the adjacency matrix - 
        nbr_bond_type: Variable(torch.LongTensor) shape (N, M)
        self_fea_idx: Node indexing  
        nbr_fea_idx: Neighbor indexing 
        ads_atom_idx: Adsorbate subgraph indexing

        Returns
        -------
        out: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution
        """

        #Assemble node properties for each main atom and its corresponding neighbors
        atom_self_fea = atom_fea[self_fea_idx, :]
        atom_nbr_fea = atom_fea[nbr_fea_idx, :]

        atom_bond_type = nbr_bond_type.view(-1,1)[nbr_fea_idx, :].double()

        #For each node cat the atom neighbor features and nbr distance
        total_fea = torch.cat([atom_nbr_fea, nbr_dist_fea],dim=1)

        total_fea = self.fc_full_1(total_fea)
        total_fea = self.bn1(total_fea)

        nbr_fea_mean = self.mean_pooling(atom_nbr_fea, self_fea_idx)

        fea_summed = self.mean_pooling(total_fea, self_fea_idx)
        node_summed = self.mean_pooling(atom_self_fea, self_fea_idx)
        
        node_summed = self.fc_full_2(node_summed)
        node_summed = self.bn2(node_summed)
        
        out = self.softplus1(node_summed + fea_summed)
        return out

class CrystalGraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """
    def __init__(self, orig_atom_fea_len, nbr_fea_dist_len, nbr_cat_value,
                 atom_fea_len=128, n_conv=3, h_fea_len=64, n_h=2, dropout_tag=0.0):
        """
        Initialize CrystalGraphConvNet.

        Parameters
        ----------

        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_dist_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        dropout_tag: float
          Dropout fraction for initiating dropout based regularisation 
        """

        super(CrystalGraphConvNet, self).__init__()
        self.mean_pooling = MeanPooling()
        
        #First embedding from original atom_fea_len to new atom len
        #Function to Apply a linear transformation to the incoming data ON THE COLUMNS
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)

        #Comvolutions from embedded layers to convolutional layers
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_dist_len=nbr_fea_dist_len)
                                    for _ in range(n_conv)])
        
        #Now its from Convolutions to NNs
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        
        #FC model 
        self.FC_model = nn.Sequential()
        for i in range(n_h - 1):
            self.FC_model.add_module('dropout_hidden'+str(i+1), nn.Dropout(p=dropout_tag))
            self.FC_model.add_module('hidden'+str(i+1), nn.Linear(h_fea_len, h_fea_len))
            self.FC_model.add_module('act'+str(i+1), nn.Softplus())

        self.FC_model.add_module('final', nn.Linear(h_fea_len, h_fea_len))

    def forward(self, atom_fea, nbr_dist_fea, nbr_adj_value, nbr_bond_type, self_fea_idx, nbr_fea_idx, ads_atom_idx, crystal_atom_idx):
        """
        Model input is arranged as per this 
        Forward pass
        
        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) 
          Atom features from atom type
        nbr_dist_fea: Variable(torch.Tensor) 
          Bond features of each atom's M neighbors
    
        nbr_adj_value: Variable(torch.LongTensor) 
          Adjacency matrix value each atom's M neighbors
    
        nbr_bond_type: Variable(torch.LongTensor) 
          Type of bonding - used for graph attention - metal-metal, metal-ads, ads-ads 
    
        self_fea_idx: torch.LongTensor 
          Indices of main nodes for each sub-graph

        nbr_fea_idx: torch.LongTensor 
          Indices of M neighbors of each atom
        
        
        crystal_atom_idx: Mapping from the crystal idx to atom idx

        Returns
        -------
        out: nn.Variable shape (N, )
          Target output 

        """
        atom_fea = self.embedding(atom_fea)
        
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_dist_fea, nbr_adj_value, nbr_bond_type, self_fea_idx, nbr_fea_idx, ads_atom_idx)

        #Pooling to create 1 X atom_fea entries for each crystal 
        crys_fea = self.mean_pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        
        # Cystal features sent to a feed-forward NN for prediction of target prop
        out = self.FC_model(crys_fea)
        return out

class MeanPooling(nn.Module):
    """
    mean pooling based from PyTorch-Scatter's mean pooling function 
    - Pooling happens based on supplemental indexing 
    More info here: https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html
    """
    def __init__(self):
        super(MeanPooling, self).__init__()
    
    def forward(self, x, index):
        mean = scatter_mean(x, index, dim=0)
        
        return mean
    
    def __repr__(self):
        return '{}'.format(self.__class__.__name__)

class SumPooling(nn.Module):
    """
    mean pooling based from PyTorch-Scatter's mean pooling function 
    - Pooling happens based on supplemental indexing 
    More info here:  https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html
    """
    def __init__(self):
        super(SumPooling, self).__init__()
    
    def forward(self, x, index):
        mean = scatter_add(x, index, dim=0)
        
        return mean
    
    def __repr__(self):
        return '{}'.format(self.__class__.__name__)
