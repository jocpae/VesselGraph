# vessap open graph benchmark dataset

import os
import os.path as osp
import sys
import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../../')
sys.path.insert(0, '../../ogb')

import torch
import numpy as np
import pandas as pd
import scipy
import argparse
import networkx as nx
import torch_geometric.transforms as T
from utils import *
from pytorch_dataset.vessap_dataset import VesselGraph
from ogb.io import DatasetSaver
from random import shuffle
import random

# step 1

# Create a constructor of DatasetSaver. dataset_name needs to follow OGB convention 
# and start from either ogbn-, ogbl-, or ogbg-. is_hetero is True for heterogeneous graphs, 
# and version indicates the dataset version.

parser = argparse.ArgumentParser(description='generate OGB Graph Prediction Dataset')
parser.add_argument('--dataset', help='Dataset name (without ogbg-).', type=str,required=True)
parser.add_argument('--no_edge_attr', action='store_true', help="whether to consider edge features in the dataset.")

args = parser.parse_args()
dataset_name = 'ogbg-' + args.dataset # e.g. ogbl-italo
if args.no_edge_attr:
    dataset_name = 'ogbg-' + args.dataset + '_no_edge_attr'
saver = DatasetSaver(dataset_name = dataset_name, is_hetero = False, version = 1)

use_edge_attr = False if args.no_edge_attr else True

# seeding for reproducible result
np.random.seed(12)

# step 2:
# Create graph_list, storing your graph objects, and call saver.save_graph_list(graph_list).
# Graph objects are dictionaries containing the following keys.

# load PyTorch Geometrics Graph

# only synthetic or vessap dataset allowed!

dataset = VesselGraph(root='data', name=args.dataset, splitting_strategy='none',use_edge_attr = use_edge_attr) # no split!
data = dataset[0]  # Get the first graph object.

print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
print(f'Contains self-loops: {data.contains_self_loops()}')
print(f'Is directed: {data.is_directed()}')

graph_list = []
num_data = len(dataset)
for i in range(len(dataset)):
    data = dataset[i]
    graph = dict()
    graph['num_nodes'] = int(data.num_nodes)
    graph['node_feat'] = np.array(data.x)
    graph['edge_index'] = np.array(data.edge_index)
    if args.no_edge_attr == False: 
        graph['edge_feat'] = np.array(data.edge_attr)
    graph_list.append(graph)

print(graph_list)
# saving a list of graphs
saver.save_graph_list(graph_list)


# step 4

# Prepare split_idx, a dictionary with three keys, train, valid, and test, and mapping into data indices of numpy.ndarray. Then, call saver.save_split(split_idx, split_name = xxx).

split_idx = dict()
num_classes = 0 

if args.dataset == 'synthetic':

    random.seed(123)
    split = np.arange(num_data)
    #split = np.random.shuffle(split)
    train_idx, val_idx,test_idx= np.array_split(split,3)
    split_idx['train'] = train_idx
    split_idx['valid'] = val_idx
    split_idx['test'] = test_idx
    labels = np.ones((num_data)).reshape(num_data,1)
    # incorrect, but necessary to satisfy save_target_labels - keep in mind, we are actually working on link prediction, not graph classification
    num_classes = 3 

elif args.dataset == 'vessap':

    split_idx['train'] = np.array([0,1,2]) # BALB
    split_idx['valid'] = np.array([3,4,5]) # Bl6J
    split_idx['test']  = np.array([6,7,8]) # CD41_E
    labels = np.array([0,0,0,1,1,1,2,2,2]).reshape(9,1)
    num_classes = 3

else:

    raise ValueError('Splitting strategy not defined for dataset name!')

saver.save_target_labels(labels)
saver.save_split(split_idx, split_name = 'type')

# step 5

# Store all the mapping information and README.md in mapping_path and call saver.copy_mapping_dir(mapping_path).
mapping_path = 'mapping/'

# prepare mapping information first and store it under this directory (empty below).
os.makedirs(mapping_path,exist_ok=True)
try:
    os.mknod(os.path.join(mapping_path, 'README.md'))
except:
    print("Readme.md already exists.")
saver.copy_mapping_dir(mapping_path)

# step 6

# Save task information by calling saver.save_task_info(task_type, eval_metric, num_classes = num_classes).
# eval_metric is used to call Evaluator (c.f. here). 
# You can reuse one of the existing metrics, or you can implement your own by creating a pull request

saver.save_task_info(task_type = 'graph classification', eval_metric = 'acc', num_classes = num_classes)

# step 7

meta_dict = saver.get_meta_dict()
print(meta_dict)

# step 7 - tesing the dataset object

from ogb.graphproppred import GraphPropPredDataset
dataset = GraphPropPredDataset(dataset_name, meta_dict = meta_dict)

# see if it is working properly
print(dataset[0])

# zip and clean up

saver.zip()
saver.cleanup()

