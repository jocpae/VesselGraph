import os
import os.path as osp
import torch
import numpy as np
import scipy
import pandas as pd
import argparse
import os
import torch_geometric.transforms as T

parser = argparse.ArgumentParser(description='display graph features and summary.')
#parser.add_argument('-d','--devices', help='Specify CUDA_VISIBLE_DEVICES.', required=True)
parser.add_argument('-ds','--dataset',help='Specify the dataset you want to select', required=True)
parser.add_argument('-s','--splitting_strategy',help='Specify the dataset you want to select', required=True)

# ensure you are not allocating wrong CUDA devices
args = parser.parse_args()
selected_dataset = args.dataset

# import PyTorch libs
from link_dataset import LinkVesselGraph

def main():

    os.system(f'rm -r $PWD/data')

    print("Testing Classes")
    dataset = LinkVesselGraph(root='data', name=selected_dataset, splitting_strategy=args.splitting_strategy, use_edge_attr=True)

    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    data = dataset[0]  # Get the first graph object.
    print(data)
    print('==============================================================')

    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
    print(f'Contains self-loops: {data.contains_self_loops()}')
    print(f'Is Undirected: {data.is_undirected()}')

    os.system(f'rm -r $PWD/data')

if __name__ == "__main__":
        main()



