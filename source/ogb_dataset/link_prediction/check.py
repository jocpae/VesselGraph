import argparse

import torch
import numpy as np
from torch_geometric.nn import Node2Vec

from ogb.linkproppred import PygLinkPropPredDataset

def main():

    dataset = PygLinkPropPredDataset(name='ogbl-vessel')
    data = dataset[0]
    split = dataset.get_edge_split()
    print(data)
    print(split)
    
    split_train_pos = split['train']['edge'].cpu().detach().numpy()
    split_train_neg = split['train']['edge_neg'].cpu().detach().numpy()
    split_test_pos = split['test']['edge'].cpu().detach().numpy()
    split_test_neg = split['test']['edge_neg'].cpu().detach().numpy()
    split_valid_pos = split['valid']['edge'].cpu().detach().numpy()
    split_valid_neg = split['valid']['edge_neg'].cpu().detach().numpy()
    # print dimensions of all

    print(split['train']['edge'].shape)
    print(split['test']['edge'].shape)
    print(split['valid']['edge'].shape)
    print(split['train']['edge_neg'].shape)
    print(split['test']['edge_neg'].shape)
    print(split['valid']['edge_neg'].shape)

    u, indices = np.unique(split_train_pos, return_index=True, axis=0)
    print("Train edge pos - unique elements", len(u))
    u, indices = np.unique(split_train_neg, return_index=True, axis=0)
    print("Train edge neg - unique elements", len(u))

    print("#####################################")

    array = np.concatenate((split_train_pos,split_train_neg,split_test_pos,split_test_neg,split_valid_pos,split_valid_neg),axis=0)
    u, indices = np.unique(array, return_index=True, axis=0)
    print("All unique elements", len(u))








if __name__ == "__main__":
    main()
