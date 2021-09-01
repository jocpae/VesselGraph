# vessap open graph benchmark dataset

import os
import os.path as osp
import sys

from numpy.random import seed

sys.path.append("../../")

import torch
import numpy as np
import pandas as pd
import scipy
import argparse
import networkx as nx
import torch_geometric.transforms as T

from pytorch_dataset.link_dataset import LinkVesselGraph
from ogb.io import DatasetSaver

# step 1

# Create a constructor of DatasetSaver. dataset_name needs to follow OGB convention 
# and start from either ogbn-, ogbl-, or ogbg-. is_hetero is True for heterogeneous graphs, 
# and version indicates the dataset version.

parser = argparse.ArgumentParser(description='generate OGB Link Prediction Dataset')
parser.add_argument('--dataset', help='Dataset name (without ogbl-).', type=str,required=True)
parser.add_argument('--use_edge_attr', action='store_true', help="whether to consider edge features in the dataset.")
parser.add_argument('--use_atlas', action='store_true', help="whether to consider Alan Brain Atlas region as node feature.")
parser.add_argument('--splitting_strategy', help='Splitting Strategy: random or spatial.', type=str,required=True)
parser.add_argument('--number_of_workers', type=str, default=4)
parser.add_argument('--seed', type=int, default=123, help="Set the seed for torch, numpy and random functions.")
parser.add_argument('--data_root_dir', type=str, default='data')

args = parser.parse_args()
dataset_name = 'ogbl-' + args.dataset + '_' + args.splitting_strategy # e.g. ogbl-BALBc_no1_spatial
if args.use_edge_attr:
    dataset_name +=  '_edge_attr' 
else:
    dataset_name +=  '_no_edge_attr' 

if args.use_atlas:
    dataset_name +=  '_atlas'


saver = DatasetSaver(dataset_name = dataset_name, is_hetero = False, version = 1)

use_edge_attr = True if args.use_edge_attr else False
use_atlas = True if args.use_atlas else False

print("Using edge attributes: ", use_edge_attr)
print("Using atlas node attributes: ", use_atlas)

# seeding for reproducible result
np.random.seed(args.seed)

# step 2:
# Create graph_list, storing your graph objects, and call saver.save_graph_list(graph_list).
# Graph objects are dictionaries containing the following keys.

# load PyTorch Geometrics Graph
dataset = LinkVesselGraph(root=os.path.join(args.data_root_dir, args.splitting_strategy), 
        name=args.dataset, splitting_strategy=args.splitting_strategy,
        number_of_workers = args.number_of_workers, seed=args.seed)

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
    graph['edge_index'] = data.edge_index.numpy() # only train pos edge index, but both directions / undirected!
    if use_edge_attr:
        graph['edge_feat'] = data.edge_attr.numpy()
    graph_list.append(graph)

print(graph_list)
# saving a list of graphs
saver.save_graph_list(graph_list)
# step 4

# Prepare split_idx, a dictionary with three keys, train, valid, and test, and mapping into data indices of numpy.ndarray. Then, call saver.save_split(split_idx, split_name = xxx)

# assign indices

# ogb only stores one directional information for the edges, so we drop every second column

split_edge = {'train': {}, 'valid': {}, 'test': {}}

# only take one direction for train edge!

split_edge['train']['edge'] = data.train_pos_edge_index.t() # these are only one directional
split_edge['train']['edge_neg'] = data.train_neg_edge_index.t() # these are only one directional
split_edge['valid']['edge'] = data.val_pos_edge_index.t() # these are only one directional
split_edge['valid']['edge_neg'] = data.val_neg_edge_index.t()  # these are only one directional
split_edge['test']['edge'] = data.test_pos_edge_index.t()  # these are only one directional
split_edge['test']['edge_neg'] = data.test_neg_edge_index.t()  # these are only one directional

if use_edge_attr:
    split_edge['train']['edge_attr'] = data.train_pos_edge_attr.t() # these are only one directional
    split_edge['test']['edge_attr'] = data.test_pos_edge_attr.t() # these are only one directional
    split_edge['valid']['edge_attr'] = data.val_pos_edge_attr.t() # these are only one directional

saver.save_split(split_edge, split_name = args.splitting_strategy)

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

saver.save_task_info(task_type = 'link prediction', eval_metric = 'acc')

# step 7

meta_dict = saver.get_meta_dict()
print(meta_dict)

# step 7 - tesing the dataset object

from ogb.linkproppred import LinkPropPredDataset
dataset = LinkPropPredDataset(dataset_name, meta_dict = meta_dict)

# see if it is working properly
print(dataset[0])
data = dataset[0]
split_edge = dataset.get_edge_split()
print(split_edge['train']['edge'].shape)
print(split_edge['test']['edge'].shape)
print(split_edge['valid']['edge'].shape)
print(split_edge['train']['edge_neg'].shape)
print(split_edge['test']['edge_neg'].shape)
print(split_edge['valid']['edge_neg'].shape)
print(data['edge_index'].shape)

# zip and clean up

saver.zip()
saver.cleanup()



