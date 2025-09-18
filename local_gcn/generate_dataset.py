"""
@author: pgg
"""
from __future__ import print_function, division

import sys
import csv
import functools
import json
import os
import random
import time
import warnings

import numpy as np
import networkx as nx

import pickle
from pymatgen.io.ase import AseAtomsAdaptor
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

from pymatgen.core.structure import Structure
from pymatgen.analysis.structure_analyzer import VoronoiConnectivity
from pymatgen.io import ase as pm_ase

from ase.io import read, write
from ase.visualize import view 
import itertools

# SurfGraph import -- import from local directory in the repo 
# SURFGRAPH_PATH = os.path.join('/'.join(os.getcwd()[:-3]), 'graph_theory_surfaces')
# sys.path.append(SURFGRAPH_PATH)
import graph_theory_surfaces.surfgraph.chemical_environment as surfgraph_chem_env


def collate_pool(dataset_list):
    """
    Custom defined collate function to assist in making the ACE-GCN dataloader
    More information about dataloaders can be found on PyTorch docs. 
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    PARAMETERS:
    ----------
    dataset_list: List. Array of individual graph-objects for various high adsorbate configurations - the graph objects are 
    GraphData class. 

    RETURNS:
    -------
    Tuple object with combined dataset of multiple high-coverage configurations
    """

    batch_atom_fea, batch_nbr_dist_fea, batch_nbr_adj, batch_nbr_bond_type = [], [], [], []
    batch_self_fea_idx, batch_nbr_fea_idx = [], []
    crystal_atom_idx, batch_target = [], []
    ads_atom_idx = []
    batch_outcar_ids = []
    base_idx = 0

    for i, ((total_atom_fea, total_node_list, total_nbr_adj_node, total_nbr_dist_node, \
        total_nbr_bond_cat, total_self_fea_idx, total_nbr_index_node), \
        target, outcar_id) in enumerate(dataset_list):
        
        #Num of adsorbates 
        num_ads = len(total_node_list)

        #Total nodes for a given coverage model 
        total_nodes = sum([len(total_node_list[i]) for i in range(0,num_ads)])

        #For every adsorbate 
        for ads_iter in range(0, num_ads):
            n_i = len(total_node_list[ads_iter])

            batch_atom_fea.append(total_atom_fea[ads_iter])
            batch_nbr_dist_fea.append(total_nbr_dist_node[ads_iter])

            batch_nbr_adj.append(total_nbr_adj_node[ads_iter])
            batch_nbr_bond_type.append(total_nbr_bond_cat[ads_iter])

            batch_self_fea_idx.extend([total_self_fea_idx[ads_iter] + base_idx])
            batch_nbr_fea_idx.extend([total_nbr_index_node[ads_iter] + base_idx])

            ads_atom_idx.extend([ ads_iter + base_idx ] * len(total_node_list[ads_iter])) #Labelling each ads in every crystal entry wrt total configs
            base_idx += n_i

        crystal_atom_idx.extend( [i] * total_nodes ) #Parent index  for every crystal entry 
        batch_target.append(target)
        batch_outcar_ids.append(outcar_id)

    return (torch.cat(batch_atom_fea, dim=0),
                torch.cat(batch_nbr_dist_fea, dim=0),
                torch.cat(batch_nbr_adj, dim=0),
                torch.cat(batch_nbr_bond_type, dim=0),

                torch.cat(batch_self_fea_idx, dim=0),
                torch.cat(batch_nbr_fea_idx, dim=0),

                torch.LongTensor(ads_atom_idx),
                torch.LongTensor(crystal_atom_idx)),\
            torch.stack(batch_target, dim=0),\
            batch_outcar_ids


def one_hot_encode_continuous(data_point, min_value=-1.0, max_value=4.0, intervals=10):
    """
    PARAMETERS: 
    ----------
    data_point = single float entry for the value to be converted to one-hot 
    min_value, max_value = bounds for the one-hot 
    interval = number of bins 
    ------------------------------------
    RETURNS: 
    One_hot encoding with dimensions 1 x intervals
    """
    if not np.isnan(data_point):
            vac_array = np.linspace(min_value, max_value, intervals)
            one_hot_vector = np.zeros((intervals,1))
            idx = (np.abs(vac_array - data_point)).argmin()
            one_hot_vector[idx,0] = 1.
    else:
            one_hot_vector = np.zeros((intervals,1))

    return one_hot_vector.T

class OHE_continuous():
    '''
    Encode a continuous variable as One-hot encoding.
    Using this instead of scikit-learn implementation due to 
    flexible end points 
    '''
    
    def __init__(self, min_value=-1.0, max_value=4.0, intervals=10):
        '''
        PARAMETERS:
        min_value, max_value = bounds for the one-hot
        interval = number of bins
        '''
        self.min_value = min_value
        self.max_value = max_value
        self.intervals = intervals
        
        self.vac_array = np.linspace(self.min_value, self.max_value, self.intervals)
        
    def transform(self, data_point):
        '''
        Apply transformation

        PARAMETERS:
        data_point = single float entry for the value to be converted to one-hot
        ------------------------------------
        RETURNS:
        One_hot encoding with dimensions 1 x intervals
        '''
        if not np.isnan(data_point):
            self.one_hot_vector = np.zeros((self.intervals,1))
            idx = (np.abs(self.vac_array - data_point)).argmin()
            self.one_hot_vector[idx,0] = 1.
            return self.one_hot_vector.T
        else:
            return self.one_hot_vector

class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.
    Think of this like a maximum likelihood type function to determine which gaussian distribtuion does 
    evaluated d_ij belong too 
    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        PARAMETERS:
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        PARAMETERS:
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp( - ((distances.reshape(-1,1) - self.filter) / self.var)**2 )


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    PARAMETERS:
    ----------

    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)

def add_to_list(graph,g_node):
    bond_distance = []
    x = []
    index = []
    for i in graph.neighbors(g_node):
        if 'ads' not in graph.nodes[i]:
            x.append(i)
            bond_distance.append(graph.edges[g_node,str(i)]['dist_edge'])
            index.append(graph.nodes[i]['index'])
    return x,bond_distance,index

def get_ads_list(graph):
    ads_nodes = []
    for n,data in graph.nodes(data='ads'):
        if data == True:
            ads_nodes.append(str(n))
    return ads_nodes

def make_adjaceny_matrix(atom_object,
                         adsorbate_elements=['C', 'O','H'],
                         radii_multiplier=1.1,
                         skin=0.25,
                         radius=1.5,
                         grid=[1,1,1],
                         look_for_Hbonds=True):
    """
    Main function to generate sub-graphs and neighbor lists 

    node_coordination -- coordination 
    `dist` categorical variable: 0 = ads-ads | 1 = ads-metal | 2 = metal-metal  
    `dist_edge` variable: actual distance 

    Implemented using Surfgraph module: 
    https://surfgraph.readthedocs.io/en/latest/Usage.html

    PARAMETERS: 
    -------------------
    atom_object: ase.object with the high coverage adsorbate configuration 
    adsorbate_elements: adsorbate elements to look for 
    radii_multiplier: number of shells to expand  c
    look_for_Hbonds: to look for H-bonds in the configuration -- append that edge as an additional edge in the subgraph

    RETURNS: 
    --------------
    total_chem_env: networkx.classes.graph.Graph  object for each ads
    total_adj: Adjacency matrix for each ads 
    (node_info, node_coordination): bond_order info and node_list coordination 

    (total_edge_dist, total_bond_cat): edge distance matrix and edge category matrix 

    """
    adsorbate_atoms = [atom.index for atom in atom_object if atom.symbol in adsorbate_elements]
    nl = surfgraph_chem_env.NeighborList(surfgraph_chem_env.natural_cutoffs(atom_object, radii_multiplier), 
                                         self_interaction=False,bothways=True, skin=skin)
    nl.update(atom_object)
    full, chem_env = surfgraph_chem_env.process_atoms(atom_object,
                                                      nl,
                                                      adsorbate_atoms,
                                                      radius=radius,
                                                      grid=grid,
                                                      H_bond=look_for_Hbonds,)

    node_information = [] #Bond order information
    node_coordination = [] #Node-wise coordination information 
    #### This for loop finds all the adsorbate surface bonds in a given graph, only for the adsorbate along which the graph is centered.
    #### Consequently it creates a bond_order for that bond, depending on the number of surface bonds detected.
    for i in range(0,len(chem_env)):
        
        node_information.append([])
        node_coordination.append([])
        
        no_surf_bonds = 0
        ads_nodes = get_ads_list(chem_env[i])
        
        for u,v,d in chem_env[i].edges(data='dist'):
            if (u in ads_nodes or v in ads_nodes) and d==1: #Calculating bond orders only for edges which have bonds 
                #print(u,v,d)
                no_surf_bonds +=1
                node_information[i].append([u,v,d])
        
        #print(no_surf_bonds)
        for j in range(0,len(node_information[i])):
            node_information[i][j][2] = 1./no_surf_bonds
        
        #Why not in adsorbate_elements? 
        for node in chem_env[i].nodes():
            node_coordination[i].append(float(len([edge for edge in full[node] if edge.split(":")[0] not in adsorbate_elements])))

    # This is the bond-order weights
    # check whether there are empty list
    node_info = np.vstack([arr for arr in node_information if len(arr) > 0])

    # This loop is for bond_order weights to the graph -- add in edge properties to each node-node interaction 
    # Edge property incorporated in the chem_env as `weight` 
    for i in range(0,len(node_info)):
        for c in range(0,len(chem_env)):
            if (node_info[i][0] in chem_env[c].nodes()) and (node_info[i][1] in chem_env[c].nodes()):
                chem_env[c].edges[node_info[i][0],node_info[i][1]]['weight'] = node_info[i][2]     
    #Creating adjacency matrix for the nodes
    adj = []
    for c in range(0,len(chem_env)):
        adj.append(nx.to_numpy_array(chem_env[c], nodelist=chem_env[c].nodes()))
    
    #Creating additional matrix based on bond type 
    edge_dist = []
    bond_category = [] 

    # edge_dist_1 = []
    # bond_category_1 = []
    for c in range(0, len(chem_env)):
        dist_node_array_each_ads = np.zeros((len(chem_env[c].nodes()), len(chem_env[c].nodes())))
        bond_type_each_ads = np.zeros((len(chem_env[c].nodes()), len(chem_env[c].nodes())))
        for k_ind, k in enumerate(chem_env[c].nodes(data=True)):
            for j_ind, j in enumerate(chem_env[c].nodes(data=True)):
                if [str(k[0]), str(j[0])] in chem_env[c].edges():
                    dist_node_array_each_ads[k_ind, j_ind] = chem_env[c].edges[str(k[0]), str(j[0])]['dist_edge']
                    bond_type_each_ads[k_ind, j_ind] = chem_env[c].edges[str(k[0]), str(j[0])]['dist']

                elif 'H_bond_O' in k[1]:
                    O_edge = 'O:{}'.format(k[1]['H_bond_O'])
                    if O_edge in str(j[0]):
                        dist_node_array_each_ads[k_ind, j_ind] = k[1]['H_bond_len']
                        dist_node_array_each_ads[j_ind, k_ind] = k[1]['H_bond_len']
                else:
                    dist_node_array_each_ads[k_ind, j_ind] = 0.0

        edge_dist.append(dist_node_array_each_ads)
        bond_category.append(bond_type_each_ads)
    # for c in range(len(chem_env)):
    #     nodes = list(chem_env[c].nodes(data=True))
    #     num_nodes = len(nodes)
    #
    #     dist_node_array_each_ads = np.zeros((num_nodes, num_nodes))
    #     bond_type_each_ads = np.zeros((num_nodes, num_nodes))
    #
    #     edges = {tuple(map(str, edge[:2])): edge[2] for edge in chem_env[c].edges(data=True)}
    #
    #     for k_ind, (k_node, k_data) in enumerate(nodes):
    #         k_str = str(k_node)
    #
    #         for j_ind, (j_node, j_data) in enumerate(nodes):
    #             j_str = str(j_node)
    #             edge_key = (k_str, j_str)
    #
    #             if edge_key in edges:
    #                 dist_node_array_each_ads[k_ind, j_ind] = edges[edge_key]['dist_edge']
    #                 bond_type_each_ads[k_ind, j_ind] = edges[edge_key]['dist']
    #             elif 'H_bond_O' in k_data:
    #                 O_edge = f"O:{k_data['H_bond_O']}"
    #                 if O_edge in j_str:
    #                     dist_node_array_each_ads[k_ind, j_ind] = k_data['H_bond_len']
    #                     dist_node_array_each_ads[j_ind, k_ind] = k_data['H_bond_len']
    #
    #     edge_dist_1.append(dist_node_array_each_ads)
    #     bond_category_1.append(bond_type_each_ads)
    return chem_env, adj, (node_info, node_coordination), (edge_dist, bond_category)

class GraphData(Dataset):
    """
    Generating graph objects for the high coverage configurations. Current iterations makes pickle files for each configuration 
    and stores them in the assigned location. If this location is mentioned in the object, the code searches for the pickle file 
    first before re-creating a new one. 

    PARAMETERS: 
    -------------------
    root_dir: Directory path for the main calculation
    id_prop_filename: Target file to parse for the configurations 
    pickle_path: Path to save pickle files 

    RETURNS: 
    -------------------
    - pkl_file for the objects 
    - graph object
    
    """
    def __init__(self,
                 structure_list,
                 label_list,file_id_list,
                 Hbonds=True,
                 dmin=0.5,
                 dmax=3.5,
                 step=0.05,
                 pickle_path=None,
                 save_pickle=True,
                 radius=1.5,
                 grid=[1,1,1],
                 return_structure=False):

        # self.root_dir = root_dir
        # assert os.path.exists(root_dir), 'root_dir does not exist!'

        # id_prop_file = os.path.join(self.root_dir, '{}'.format(id_prop_filename))
        # assert os.path.exists(id_prop_file), 'ID Prop file does not exist!'
        #
        # with open(id_prop_file) as f:
        #     reader = csv.reader(f)
        #     self.id_prop_data = [row for row in reader]
        self.id_prop_data = label_list
        self.file_id_list = file_id_list
        
        atom_init_file = os.path.join("./", 'atom_init.json')

        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=dmax, step=step)
        
        self.save_pickle = save_pickle

        if pickle_path is None:
            self.pickle_path = "./data/pickle"
            if not os.path.exists(self.pickle_path):
                os.mkdir(self.pickle_path)
        else:
            self.pickle_path = pickle_path
        print('Pickle path: {}'.format(self.pickle_path))
        self.structure_list = structure_list
        self.Hbonds = Hbonds
        self.radius = radius
        self.grid = grid
        self.return_structure = return_structure

    def __len__(self):
        return len(self.id_prop_data)

    def view_atom_object(self, idx):
        outcar_id, _, _ = self.id_prop_data[idx]
        return view(read(os.path.join(self.root_dir, outcar_id)))

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        file_id = self.file_id_list[idx]
        target = self.id_prop_data[idx]
        bridge = pm_ase.AseAtomsAdaptor()
        atom_object = bridge.get_atoms(self.structure_list[idx])
        x_pymat=bridge.get_structure(atom_object)
        try:
            with open('{}/{}.pkl'.format(self.pickle_path, file_id), 'rb') as f:
                outcar_data = pickle.load(f)
                (total_atom_fea, total_node_list, total_nbr_adj_node, total_nbr_dist_node, total_nbr_bond_cat, total_self_fea_idx, total_nbr_index_node), target, poscar_id = outcar_data
                if self.return_structure:
                    return (total_atom_fea, total_node_list, total_nbr_adj_node, total_nbr_dist_node, total_nbr_bond_cat,
                     total_self_fea_idx, total_nbr_index_node), target, poscar_id, x_pymat
                return (total_atom_fea, total_node_list, total_nbr_adj_node, total_nbr_dist_node, total_nbr_bond_cat,
                        total_self_fea_idx, total_nbr_index_node), target, poscar_id

        except FileNotFoundError:
            print('{}/{}.pkl not found, parsing outcar'.format(self.pickle_path, file_id))

        total_chem_env, total_adj, (node_info, node_coordination), (total_edge_dist, total_bond_cat) = make_adjaceny_matrix(atom_object,radius=self.radius,
                                                                                                                            grid=self.grid,
                                                                                                                            look_for_Hbonds=self.Hbonds)
        #print(len(node_coordination[0]))

        total_node_list = [list(total_chem_env[ads_iter].nodes) for ads_iter in range(len(total_chem_env))]
        total_atom_fea = []

        #Append atom elemental featuere and one-hot CN information
        for ads_num, node_list in enumerate(total_node_list):
            atom_indices = [int(node_list[i].split(':')[-1].split('[')[0]) for i in range(len(node_list))]
            atom_fea = np.vstack([self.ari.get_atom_fea(x_pymat[i].specie.number) for i in atom_indices])
            cn_mat = np.vstack([one_hot_encode_continuous(node_cn, min_value=0,max_value=12,intervals=12) for node_cn in node_coordination[ads_num]])
            atom_fea_combined = np.concatenate((atom_fea, cn_mat), axis=1)
            atom_fea_combined = torch.Tensor(atom_fea_combined)
            total_atom_fea.append(atom_fea_combined)

        total_nbr_index_node = [] #Index of the neighbors
        total_nbr_adj_node = [] # Node-neighbor adj matrix value
        total_nbr_bond_cat = [] #Node-neighbor bond category
        total_nbr_dist_node = [] #Nbr distance
        total_self_fea_idx = [] #Node atom index

        #For every adsorbate and the corresponding graph it has
        for ads_iter in range(len(total_chem_env)):
            ads_chem_env = total_chem_env[ads_iter] #Graph object for each adsorbate
            ads_adj = total_adj[ads_iter] #Adjacnecy matrix for each adsorbate
            ads_dist_mat = total_edge_dist[ads_iter] #Node-neighbor distance for each adsorbate
            ads_bond_category = total_bond_cat[ads_iter] #Node-neighbor bond category
            # node_list = total_node_list[ads_iter] #Node list for each adsorbate sub-graph

            nbr_index_node = [] # What are the nbr index
            nbr_adj_node = [] # Node-neighbor adj entry
            nbr_dist_node = [] # Node-neighbor distance entry
            nbr_dist_category = [] # Node-neighbor bond category entry
            self_fea_idx = []

            #For every node in that graph
            for node_ind in range(len(ads_chem_env)):
                nbr_idx, nbr_connection, nbr_bond_cat, nbr_dist = [], [], [], []
                #Use the adjacency matrix to make the nbr_dist array for every node in the graph -- nbr_idx = neighbor indice; nbr_fea_idx = nbr_dist for those edges
                for i, nbr_adj_entry in enumerate(ads_adj[node_ind]):
                    if np.round(nbr_adj_entry,1) > 0.0: #Append only those neighbors which are connected
                        nbr_idx.append(i) #To get atom index wrt original sub-graph
                        nbr_connection.append(nbr_adj_entry)
                        nbr_dist.append(ads_dist_mat[node_ind][i])
                        nbr_bond_cat.append(ads_bond_category[node_ind][i])

                nbr_index_node.extend(nbr_idx)
                nbr_adj_node.extend(nbr_connection)

                nbr_dist_node.extend(nbr_dist)
                nbr_dist_category.extend(nbr_bond_cat)

                # Main index vector to higlight the head nodes for each neighbors
                # self_fea_idx is made based on the number of neighbor each node has
                self_fea_idx.extend( [node_ind] * len(nbr_idx) )

            #Transform the features to tensors
            nbr_adj_node = torch.Tensor(np.array(nbr_adj_node))


            #nbr_dist_node = torch.Tensor(self.gdf.expand(np.array(nbr_dist_node)))
            # nbr_dist_node_list = torch.Tensor(np.array(nbr_dist_node))
            nbr_dist_node = torch.Tensor(np.vstack([one_hot_encode_continuous(node_edge, min_value=0.5,max_value=3.5,intervals=6) for node_edge in nbr_dist_node]))

            nbr_dist_category = torch.Tensor(np.array(nbr_dist_category))
            nbr_index_node = torch.LongTensor(np.array(nbr_index_node))
            self_fea_idx = torch.LongTensor(self_fea_idx)


            #Appending it to the big array for the entire POSCAR
            total_self_fea_idx.append(self_fea_idx)
            total_nbr_index_node.append(nbr_index_node)

            total_nbr_adj_node.append(nbr_adj_node)
            total_nbr_dist_node.append(nbr_dist_node)
            total_nbr_bond_cat.append(nbr_dist_category)

        target = torch.Tensor([float(target)])

        #### TODO test: adding bond features to atom features ####
        # for idx, _ in enumerate(total_bond_cat):
        #     total_atom_fea[idx] = torch.cat((total_atom_fea[idx], total_bond_cat[idx]), dim=-1)
        ##########################################################

        if self.save_pickle:
            print('Saving pickle {}'.format(file_id))
            with open('{}/{}.pkl'.format(self.pickle_path, file_id), 'wb') as f:
                outcar_prop = (total_atom_fea, total_node_list, total_nbr_adj_node, total_nbr_dist_node, total_nbr_bond_cat, total_self_fea_idx, total_nbr_index_node), target, file_id
                pickle.dump(outcar_prop, f)
        if self.return_structure:
            return (total_atom_fea, total_node_list, total_nbr_adj_node, total_nbr_dist_node, total_nbr_bond_cat, total_self_fea_idx, total_nbr_index_node), target, file_id, x_pymat
        return (total_atom_fea, total_node_list, total_nbr_adj_node, total_nbr_dist_node, total_nbr_bond_cat, total_self_fea_idx, total_nbr_index_node), target, file_id
