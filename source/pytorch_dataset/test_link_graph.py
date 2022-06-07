import os
import os.path as osp
import torch
import numpy as np
import scipy
import pandas as pd
import argparse
import os
import torch_geometric.transforms as T
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data

parser = argparse.ArgumentParser(description='display graph features and summary.')
parser.add_argument('-ds','--dataset',help='Specify the dataset you want to select', required=True)
parser.add_argument('-s','--splitting_strategy',help='Specify the dataset you want to select', required=True)

args = parser.parse_args()
selected_dataset = args.dataset

# import PyTorch libs
from link_dataset import LinkVesselGraph

def main():

    dataset = LinkVesselGraph(root='data', name=selected_dataset, splitting_strategy=args.splitting_strategy, use_edge_attr=True, use_atlas=True)

    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    data = dataset[0]  # Get the first graph object.
    print(data)
    print('==============================================================')

    # Gather some statistics about the graph.
    print(f'Number of nodes in graph: {data.num_nodes}')
    print(f'Number of edges in graph: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
    print(f'Contains self-loops: {data.contains_self_loops()}')
    print(f'Is Undirected: {data.is_undirected()}')

    print(f'Number of undirected edges', data.edge_index_undirected.size(dim=1))
    print(f'Number of training edges', data.train_pos_edge_index.size(dim=1))
    print(f'Number of validation edges', data.val_pos_edge_index.size(dim=1))
    print(f'Number of test edges', data.test_pos_edge_index.size(dim=1))

    # Caution: if you would like to convert all edges to networkx graph, please
    # overwrite data.edge_index with data.edge_index_undirected. 
    # The link dataset adheres to the convention that only training edges are 
    # present in the data.edge_index. However, to obtain the full graph, we have to pass
    # all edges to the networkx function.

    data_undirected = Data(x=data.x, edge_index = data.edge_index_undirected,
                      edge_attr = data.edge_attr_undirected)

    G = to_networkx (data_undirected, to_undirected=False)
    print("Networkx: #nodes, #edges", G.number_of_nodes(), G.number_of_edges())







if __name__ == "__main__":
        main()



