import pandas as pd
import torch
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import os
import random
from torch_geometric.utils import to_undirected, remove_isolated_nodes, remove_self_loops, train_test_split_edges, negative_sampling

import math
# our modified example
def custom_train_test_split_edges(data, val_ratio: float = 0.05, test_ratio: float = 0.1):
    r"""Splits the edges of a :class:`torch_geometric.data.Data` object
    into positive and negative train/val/test edges.
    As such, it will replace the :obj:`edge_index` attribute with
    :obj:`train_pos_edge_index`, :obj:`train_pos_neg_adj_mask`,
    :obj:`val_pos_edge_index`, :obj:`val_neg_edge_index` and
    :obj:`test_pos_edge_index` attributes.
    If :obj:`data` has edge features named :obj:`edge_attr`, then
    :obj:`train_pos_edge_attr`, :obj:`val_pos_edge_attr` and
    :obj:`test_pos_edge_attr` will be added as well.

    Args:
        data (Data): The data object.
        val_ratio (float, optional): The ratio of positive validation edges.
            (default: :obj:`0.05`)
        test_ratio (float, optional): The ratio of positive test edges.
            (default: :obj:`0.1`)

    :rtype: :class:`torch_geometric.data.Data`
    """

    assert 'batch' not in data  # No batch-mode.

    num_nodes = data.num_nodes
    row, col = data.edge_index
    edge_attr = data.edge_attr
    data.edge_index = data.edge_attr = None

    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]
    if edge_attr is not None:
        edge_attr = edge_attr[perm]

    r, c = row[:n_v], col[:n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)
    if edge_attr is not None:
        data.val_pos_edge_attr = edge_attr[:n_v]

    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)
    if edge_attr is not None:
        data.test_pos_edge_attr = edge_attr[n_v:n_v + n_t]

    r, c = row[n_v + n_t:], col[n_v + n_t:]

    # this section is custom
    # -----------------------
    data.train_pos_edge_index = torch.stack([r, c], dim=0)

    helper = data.train_pos_edge_index

    if edge_attr is not None:
        out = to_undirected(data.train_pos_edge_index, edge_attr[n_v + n_t:])
        data.edge_index, data.edge_attr = out
    else:
        data.edge_index = to_undirected(data.train_pos_edge_index)

    data.train_pos_edge_index = helper
    # -----------------------

    # generate negative edge list by randomly sampling the nodes!
    neg_edge_list = np.array(np.random.randint(low=0, high=num_nodes,
                                               size=(2*data.edge_index_undirected.shape[1],)). # left and right edge - 2x, to be safe:3.4
                             reshape((data.edge_index_undirected.shape[1],2)))

    a = np.min(neg_edge_list, axis=1)
    b = np.max(neg_edge_list, axis=1)

    neg_edge_list = np.vstack((a,b)).transpose()

    # filter for unique edges in the negative edge list

    # obtain the indexes of the first occuring objects
    # _, indices = np.unique(edges[:,[0,1]],return_index=True,axis=0)
    _, indices = np.unique(neg_edge_list[:,[0,1]],return_index=True,axis=0)

    neg_edge_list = neg_edge_list[indices]

    all_edges = np.concatenate((np.array(data.edge_index_undirected.t()),neg_edge_list), axis=0) # concat positive edges of graph and negative edges

    # obtain the indexes of unique objects
    _, indices = np.unique(all_edges[:, [0, 1]], return_index=True, axis=0)

    # sort indices

    indices = np.sort(indices)
    indices = indices[indices > data.edge_index_undirected.shape[1]] # remove the indices of the positive edges!
    neg_edge_list = torch.tensor(all_edges[indices])

    # sample edges according to percentage

    ind = torch.randperm(neg_edge_list.shape[0])

    data.val_neg_edge_index = neg_edge_list[ind[:n_v]].t()
    data.test_neg_edge_index = neg_edge_list[ind[n_v:n_v+n_t]].t()
    data.train_neg_edge_index = neg_edge_list[ind[n_v+n_t:n_v+n_t+data.train_pos_edge_index.shape[1]]].t()

    """
    #Original Sampling: allocates to much memory

    # Negative edges.
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0

    neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
    ind = torch.randperm(neg_row.size(0))
    perm = ind[:n_v + n_t]
    perm_train = ind[n_v+n_t:n_v+n_t+data.train_pos_edge_index.shape[1]]
    neg_row_train, neg_col_train = neg_row[perm_train], neg_col[perm_train]
    neg_row, neg_col = neg_row[perm], neg_col[perm]

    neg_adj_mask[neg_row, neg_col] = 0
    data.train_neg_adj_mask = neg_adj_mask

    row, col = neg_row[:n_v], neg_col[:n_v]
    data.val_neg_edge_index = torch.stack([row, col], dim=0)

    row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
    data.test_neg_edge_index = torch.stack([row, col], dim=0)

    row, col = neg_row_train , neg_col_train
    data.train_neg_edge_index = torch.stack([row, col], dim=0)
    """

    return data

# spatial sampling of positive links does not make any sense at all!

def negative_sampling(all_undirectional_edges, df_edges, data, set_pos_edge,num_samples):

    # seed random functions
    np.random.seed(123) # init only once (not in loop)
    torch.manual_seed(123)
    random.seed(123)

    # determine spatial criteria
    mean = df_edges["distance"].mean()
    delta = mean
    median = df_edges["distance"].median()

    # contains all possible node ids of the set
    set_nodes = np.unique(set_pos_edge.numpy().flatten()) # contains all possible nodes of the set, unique as edges are undirected
    set_nodes = np.column_stack((set_nodes,np.array(data.pos[set_nodes][:])))

    edge_list = []
    edges = np.array(all_undirectional_edges,dtype=int)

    print(edges.shape)

    with tqdm(total=int(num_samples) )as pbar:
        while len(edge_list) < int(num_samples):

            num_nodes = len(set_nodes)

            #  fetch a random node that is contained in the set
            source_node_idx = int(np.random.randint(num_nodes,size=1))
            
            # get X/Y/Z of random node
            x_centre = set_nodes[source_node_idx,1]
            y_centre = set_nodes[source_node_idx,2]
            z_centre = set_nodes[source_node_idx,3]

            # create filter conditions
            x_min = x_centre - delta
            x_max = x_centre + delta
            y_min = y_centre - delta
            y_max = y_centre + delta
            z_min = z_centre - delta
            z_max = z_centre + delta

            temp = set_nodes # start with complete set
            temp = temp[(temp[:,1] <= x_max) & (temp[:,1] >= x_min)]
            temp = temp[(temp[:,2] <= y_max) & (temp[:,2] >= y_min)]
            temp = temp[(temp[:,3] <= z_max) & (temp[:,3] >= z_min)]

            # drop the current node to omit self-nodes
            temp = temp[(temp[:,0]!=set_nodes[source_node_idx,0])]

            if temp.shape[0] > 0:

                source_node_id = int(set_nodes[source_node_idx,0])
                target_node_index = int(np.random.randint(low=0,high=len(temp[:,0]),size=1))
                target_node_id = int(temp[target_node_index,0])# get node ID (not row ID)

                # check if edge exists in data.edge_index if so discard the sample
                if any((edges.transpose()[:]==[target_node_id,source_node_id]).all(1)) == False: # one direction is sufficient

                    if len(edge_list) > 0: # othewrsie edge_list empty
                        # check if edge exists in negative samples list, if so discard the sample
                        if any((np.array(edge_list[:])==[target_node_id,source_node_id]).all(1)) == False:
                            # add directed edges
                            edge_list.append([source_node_id,target_node_id]) 
                            #edge_list.append([target_node_id,source_node_id])
                            pbar.update(1)

                    else:
                        # add undirected edges
                        edge_list.append([source_node_id,target_node_id]) 
                        #edge_list.append([target_node_id,source_node_id])
                        pbar.update(1)
 
    return torch.from_numpy(np.array(edge_list).T)

