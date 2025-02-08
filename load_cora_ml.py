from torch_geometric.datasets import CitationFull
import torch
from copy import deepcopy
from torch_geometric.data import InMemoryDataset, extract_zip, Data
from torch_geometric.utils import degree, subgraph, to_undirected, to_dense_adj
import random
import os
import os.path as osp
import itertools
import gdown
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset
from scipy.sparse import csr_matrix
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utils import *



class DataInfo(object):
    r"""
    The class for data point storage. This enables tackling node data point like graph data point, facilitating data splits.
    """
    def __init__(self, idx, y):
        super(DataInfo, self).__init__()
        self.storage = []
        self.idx = idx
        self.y = y

    def __repr__(self):
        s = [f'{key}={self.__getattribute__(key)}' for key in self.storage]
        s = ', '.join(s)
        return f"DataInfo({s})"

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if key != 'storage':
            self.storage.append(key)




class DomainGetter(object):
    r"""
    A class containing methods for data domain extraction.
    """
    def __init__(self):
        pass

    def get_degree(self, graph: Data) -> int:
        """
        Args:
            graph (Data): The PyG Data object.
        Returns:
            The degrees of the given graph.
        """
        try:
            node_degree = degree(graph.edge_index[0], graph.num_nodes)
            return node_degree
        except ValueError as e:
            print('#E#Get degree error.')
            raise e

    def get_word(self, graph: Data) -> int:
        """
        Args:
            graph (Data): The PyG Data object.
        Returns:
            The word diversity value of the graph.
        """
        num_word = graph.x.sum(1)
        return num_word


def get_domain_sorted_indices(num_data, graph, domain='degree'):
    domain_getter = DomainGetter()
    graph.__setattr__(domain, getattr(domain_getter, f'get_{domain}')(graph))

    data_list = []
    for i in range(num_data):
        data_info = DataInfo(idx=i, y=graph.y[i])
        data_info.__setattr__(domain, graph.__getattr__(domain)[i])
        data_list.append(data_info)

    sorted_data_list = sorted(data_list, key=lambda data: getattr(data, domain))

    # Assign domain id
    cur_domain_id = -1
    cur_domain = None
    sorted_domain_split_data_list = []
    for data in sorted_data_list:
        if getattr(data, domain) != cur_domain:
            cur_domain = getattr(data, domain)
            cur_domain_id += 1
            sorted_domain_split_data_list.append([])
        data.domain_id = torch.LongTensor([cur_domain_id])
        sorted_domain_split_data_list[data.domain_id].append(data)
        # print("data.domain_id",data.domain_id)
        

    return sorted_data_list, sorted_domain_split_data_list

def extract_nodes(graph, sorted_domain_split_data_list, domain_range, label_range, train_ratio=0.8):
 
    valid_node_ids = []
    for domain_id in range(domain_range[0], domain_range[1] + 1):
     
        if domain_id < len(sorted_domain_split_data_list):
            domain_data_list = sorted_domain_split_data_list[domain_id]
   
            for data in domain_data_list:
                if label_range[0] <= data.y <= label_range[1]: 
                    valid_node_ids.append(data.idx)
    for domain_id in range(90, 139):
   
        if domain_id < len(sorted_domain_split_data_list):
            domain_data_list = sorted_domain_split_data_list[domain_id]
           
            for data in domain_data_list:
                if label_range[0] <= data.y <= label_range[1]: 
                    valid_node_ids.append(data.idx)
    label = graph.y 
   
    node_feat = graph.x 
    

    
    train_node_ids, val_node_ids = train_test_split(valid_node_ids, train_size=train_ratio, random_state=42)
    train_node_ids=torch.tensor(train_node_ids)
    val_node_ids=torch.tensor(val_node_ids)
    
    
    train_feats = node_feat[train_node_ids] 
    val_feats = node_feat[val_node_ids]
    train_labels = label[train_node_ids]
    val_labels = label[val_node_ids]
    
    edge_index = graph.edge_index
    
    
    train_edge_mask = (
        torch.isin(edge_index[0], train_node_ids) &
        torch.isin(edge_index[1], train_node_ids)
    )
    val_edge_mask = (
        torch.isin(edge_index[0], val_node_ids) &
        torch.isin(edge_index[1], val_node_ids)
    )

   
    train_edge_index = edge_index[:, train_edge_mask]
    val_edge_index = edge_index[:, val_edge_mask]
    
    train_node_mapping = {node_id.item(): idx for idx, node_id in enumerate(train_node_ids)}
    print("train_node_mapping", train_node_mapping)
    
    mapped_train_edge_index = torch.tensor([train_node_mapping[node.item()] for node in train_edge_index.flatten()]).view(2, -1)
    
    
    val_node_mapping = {node_id.item(): idx for idx, node_id in enumerate(val_node_ids)}
    print("val_node_mapping", val_node_mapping)
    
    mapped_val_edge_index = torch.tensor([val_node_mapping[node.item()] for node in val_edge_index.flatten()]).view(2, -1)
    
    
    
    
    if torch.any(mapped_train_edge_index >= len(train_node_ids)):
        raise ValueError(f"Some node indices in edge_index are out of bounds. Maximum node index: {len(train_node_ids) - 1}")
    
    val_mapping = torch.range(0,val_feats.shape[0]-1)
    
    ids_per_cls = get_class_by_id(label)
    
    
    np.savez_compressed(
    './dataset/cora_ml_stream/cora_ml_2base_stream.npz',
    base_train_x_feature=train_feats.numpy(),
    base_train_y_label=train_labels.numpy(),
    base_train_edge_index=mapped_train_edge_index.numpy(),
    base_val_x_feature=val_feats.numpy(),
    base_val_y_label=val_labels.numpy(),
    base_val_edge_index=mapped_val_edge_index.numpy(),
    val_mapping_idx=val_mapping.numpy(),
    train_id = train_node_ids.numpy(),
    test_id = val_node_ids.numpy(),
    ids_per_cls = ids_per_cls)
    

    npz_file = np.load('./dataset/cora_ml_stream/cora_ml_2base_stream.npz')
    
    x_train, y_train, edge_index_train = torch.tensor(npz_file['base_train_x_feature']), torch.tensor(npz_file['base_train_y_label']), torch.tensor(npz_file['base_train_edge_index'])#
    train_class_by_id = get_class_by_id(y_train) #
    train_node_num = y_train.shape[0] #
    
    x_val, y_val, edge_index_val, val_mapping_idx = torch.tensor(npz_file['base_val_x_feature']), torch.tensor(npz_file['base_val_y_label']), torch.tensor(npz_file['base_val_edge_index']), npz_file['val_mapping_idx']
    val_class_by_id = get_class_by_id(y_val)
    
    
    print("train_node_num: ", train_node_num)
    print("train adj",edge_index_train)
    
    return torch.tensor(valid_node_ids)
    

def load_cora_stream(graph, base_node, sorted_domain_split_data_list, stream_num, domain_range, label_range, train_ratio=0.8):
   
    valid_node_ids = []
    for domain_id in range(domain_range[0], domain_range[1] + 1):
       
        if domain_id < len(sorted_domain_split_data_list):
            domain_data_list = sorted_domain_split_data_list[domain_id]
            
            for data in domain_data_list:
                if label_range[0] <= data.y <= label_range[1]: 
                    valid_node_ids.append(data.idx)
    
    label = graph.y 
    
    node_feat = graph.x 
    

    
    valid_node_ids = torch.tensor(valid_node_ids)
    valid_node_ids=torch.cat((base_node, valid_node_ids), 0)
    
    train_node_ids, test_node_ids = train_test_split(valid_node_ids, train_size=train_ratio, random_state=42)
    train_node_ids=torch.tensor(train_node_ids)
    test_node_ids=torch.tensor(test_node_ids)
    
    
    train_feats = node_feat[train_node_ids] 
    test_feats = node_feat[test_node_ids] 
    train_labels = label[train_node_ids]
    test_labels = label[test_node_ids]
    
    edge_index = graph.edge_index

    
    train_edge_mask = (
        torch.isin(edge_index[0], train_node_ids) &
        torch.isin(edge_index[1], train_node_ids)
    )
    test_edge_mask = (
        torch.isin(edge_index[0], test_node_ids) &
        torch.isin(edge_index[1], test_node_ids)
    )

   
    train_edge_index = edge_index[:, train_edge_mask]
    test_edge_index = edge_index[:, test_edge_mask]
    
    train_node_mapping = {node_id.item(): idx for idx, node_id in enumerate(train_node_ids)}
   
    
    mapped_train_edge_index = torch.tensor([train_node_mapping[node.item()] for node in train_edge_index.flatten()]).view(2, -1)
    
    
    test_node_mapping = {node_id.item(): idx for idx, node_id in enumerate(test_node_ids)}

    
    mapped_test_edge_index = torch.tensor([test_node_mapping[node.item()] for node in test_edge_index.flatten()]).view(2, -1)
    

    
    if torch.any(mapped_train_edge_index >= len(train_node_ids)):
        raise ValueError(f"Some node indices in edge_index are out of bounds. Maximum node index: {len(train_node_ids) - 1}")
    
    ids_per_cls = get_class_by_id(train_labels)
    ft_mapping = torch.arange(0, int(train_feats.shape[0] * 0.3))
    test_mapping = torch.arange(0, int(test_feats.shape[0] * 0.3))
    np.savez_compressed(
    './dataset/cora_ml_stream/cora_ml_2base_{}idx_1novel_stream.npz'.format(stream_num),
    ft_x_feature=train_feats.numpy(),
    ft_y_label=train_labels.numpy(),
    ft_edge_index=mapped_train_edge_index.numpy(),
    test_x_feature=test_feats.numpy(),
    test_y_label=test_labels.numpy(),
    test_edge_index=mapped_test_edge_index.numpy(),
    ft_mapping_idx=ft_mapping.numpy(),
    test_mapping_idx=test_mapping.numpy(),
    train_id = train_node_ids.numpy(),
    test_id = test_node_ids.numpy(),
    ids_per_cls = ids_per_cls)
    

    
    
    return torch.tensor(valid_node_ids)
    
dataset = CitationFull(root='./', name='Cora_ML')
graph = dataset[0]
num_sample = graph.x.shape[0]
print(graph)


dataname = 'CoraML'
datatype = 'word'

word_list=[]

sorted_data_list, sorted_domain_split_data_list = get_domain_sorted_indices(num_sample, graph, datatype)


for i in range(len(sorted_domain_split_data_list)):
    print("domain_id",i,"node_num",len(sorted_domain_split_data_list[i]))


domain_range = [0,139]
label_range = [0,1]
base_node_id = extract_nodes(graph, sorted_domain_split_data_list, domain_range, label_range, train_ratio=0.8)
domain_list = [[40,80], [61,100], [60,100], [50,110], [0,139]]
label_list = [[2,2], [3,3], [4,4],[5,5],[6,6]]
stream=5
for i in range (1, stream+1):
    domain_range = domain_list[i-1]
    label_range = label_list[i-1]
    
    node_stream = load_cora_stream(graph, base_node_id, sorted_domain_split_data_list, i, domain_range, label_range, train_ratio=0.7)
    base_node_id = node_stream
 
    print(base_node_id.shape)











