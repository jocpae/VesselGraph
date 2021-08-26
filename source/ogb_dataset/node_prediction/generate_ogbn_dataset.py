import os
import os.path as osp
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
from pytorch_dataset.node_dataset import NodeVesselGraph
from pytorch_dataset.vessap_utils import *
from ogb.io import DatasetSaver

# for multi-class labeling
from sklearn.preprocessing import KBinsDiscretizer

# step 1

# Create a constructor of DatasetSaver. dataset_name needs to follow OGB convention 
# and start from either ogbn-, ogbl-, or ogbg-. is_hetero is True for heterogeneous graphs, 
# and version indicates the dataset version.

parser = argparse.ArgumentParser(description='generate OGB Node Prediction Dataset')
parser.add_argument('-ds','--dataset', help='Dataset name (without ogbn-).', type=str,required=True)
args = vars(parser.parse_args())
ds_name = args['dataset']

dataset_name = 'ogbn-' + ds_name # e.g. ogbl-italo
saver = DatasetSaver(dataset_name = dataset_name, is_hetero = False, version = 1)

dataset = NodeVesselGraph(root='data', name=ds_name, pre_transform=T.LineGraph(force_directed=False))
data = dataset[0]  # Get the first graph object.
graph_list = []
graph = dict()

# Define Index
# assign id to edge_attr_keys
indexed_edge_attr_keys = [f'{i}: {c}' for i, c in enumerate(data.edge_attr_keys)]
print('Available Features: \n ', "\n  ".join(indexed_edge_attr_keys))
desired_edge_attr_indices = np.array(input("Enter indices of desired features (Use \",\" to separate them): ").split(",")).astype(np.int)
index = int(input("Enter feature index of desired label: "))
print(f"Dataset is created with the following features: \n   ",
      '\n  '.join(np.take(data.edge_attr_keys, desired_edge_attr_indices)),
      f"\nThe utilized label is: \n  {data.edge_attr_keys[index]}")

# Create Labels
class_type = input(f"Choose between a certain number of balanced classes (bc) or define classes by pixel boundaries (pb): ")
class_type = class_type.lower()
labels = np.array(data.x)
labels = labels[:, index].reshape((len(labels), 1)) # 0 for length, 1 for distance, ....

if class_type == 'bc':
    num_classes = int(input("Enter number of desired class: ")) # choose wisely, this is a random guess
    est = KBinsDiscretizer(n_bins=num_classes, encode='ordinal', strategy='quantile')#approximately the same number of samples, balanced set!
    labels = est.fit_transform(labels)
    dataset_name = dataset_name + '_balanced_classes'
elif class_type == 'pb':
    classes = np.zeros(labels.shape)
    class_boundaries = np.array(input("Enter desired radius boundaries as pixel values (Use \",\" to separate them): ").split(",")).astype(np.float)
    for class_id, boundary in enumerate(class_boundaries, 1):
        classes[labels > boundary] = class_id
    labels = classes
    dataset_name = dataset_name + '_boundary_classes'

# step 2:
# Create graph_list, storing your graph objects, and call saver.save_graph_list(graph_list).
# Graph objects are dictionaries containing the following keys.

# fill dict
graph['num_nodes'] = int(data.num_nodes)
graph['node_feat'] = np.take(np.array(data.x), desired_edge_attr_indices, axis=1) # axis = 1 is column!
graph['edge_index'] = np.array(data.edge_index)

# saving a list of graphs
graph_list.append(graph)
saver.save_graph_list(graph_list)
saver.save_target_labels(labels)

# step 4
# Prepare split_idx, a dictionary with three keys, train, valid, and test, and mapping into data indices of numpy.ndarray. Then, call saver.save_split(split_idx, split_name = xxx)

split_idx = dict()
num_data = len(labels)
np.random.seed(12)
perm = np.random.permutation(num_data)
split_idx['train'] = torch.from_numpy(perm[:int(0.8*num_data)])
split_idx['valid'] = torch.from_numpy(perm[int(0.8*num_data): int(0.9*num_data)])
split_idx['test'] = torch.from_numpy(perm[int(0.9*num_data):])
saver.save_split(split_idx, split_name = 'random')

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
saver.save_task_info(task_type = 'multiclass classification', eval_metric = 'acc', num_classes = len(class_boundaries + 1))

# step 7

meta_dict = saver.get_meta_dict()

# step 7 - tesing the dataset object

from ogb.nodeproppred import NodePropPredDataset
dataset = NodePropPredDataset(dataset_name, meta_dict = meta_dict)

# see if it is working properly
# print(dataset[0])

# zip and clean up
saver.zip()
saver.cleanup()

# copy submission directory to datasets

