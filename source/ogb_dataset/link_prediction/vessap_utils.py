import pandas as pd
import torch
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import os

def get_pos_edge_split(data, val_ratio=0.05, test_ratio = 0.1,use_edge_attr=True):
   
    # positive sampling
    # [node1id, node2id]
    edge_coord = data.edge_index
    # [node1id, x, y, z]
    node1_coord = data.pos[data.edge_index[0,:]]
    # [node2id, x, y, z]
    node2_coord = data.pos[data.edge_index[1,:]]
    # [x_centre, y_centre, z_centre] 
    edge_ctr = 0.5 *(node1_coord + node2_coord)

    min_ratio = min([val_ratio,test_ratio])
    q = int(1/min_ratio)
    train_label = q - int(test_ratio / min_ratio) - int(val_ratio / val_ratio)
    val_label = q - int(val_ratio / val_ratio)
    test_label = q

    # tensor binning train - val - test 
    df_bins = pd.DataFrame({'x': edge_ctr[:, 0], 'y': edge_ctr[:, 1],'z': edge_ctr[:, 1]})
    bins = np.array(pd.qcut(df_bins['x'], q=q, labels=False))

    # split according to bins
    data.train_pos_edge_index = torch.from_numpy(np.argwhere(bins < train_label ).flatten())
    data.val_pos_edge_index = torch.from_numpy(np.intersect1d(np.argwhere(bins >=train_label) , np.argwhere(bins<val_label)).flatten())
    data.test_pos_edge_index = torch.from_numpy(np.argwhere(bins >=val_label).flatten())

    # assign edges
    data.test_pos_edge = data.edge_index[:,data.test_pos_edge_index]
    data.train_pos_edge = data.edge_index[:,data.train_pos_edge_index]
    data.val_pos_edge = data.edge_index[:,data.val_pos_edge_index]

    if use_edge_attr:

        # edge features
        data.test_pos_edge_attr = data.edge_attr[data.test_pos_edge_index,:]
        data.train_pos_edge_attr = data.edge_attr[data.train_pos_edge_index,:]
        data.val_pos_edge_attr = data.edge_attr[data.val_pos_edge_index,:]

    return data

def negative_sampling(df_edges,df_nodes,data,set_pos_edge,set_pos_edge_index):

    mean = df_edges["length"].mean()
    sigma = df_edges["length"].std()
    delta = mean #+ sigma

    # contains all possible node ids of the set
    set_nodes = np.unique(set_pos_edge.numpy().flatten()) # contains all possible nodes of the set, unique as edges are undirected
    set_nodes = np.column_stack((set_nodes,np.array(data.pos[set_nodes][:])))

    edge_list = []#np.zeros((2,len(set_pos_edge_index)))
    edges = np.array(data.edge_index,dtype=int)

    #for i in tqdm(range(int(leset_pos_edge_indexn(set_pos_edge_index)/2))): # didvided by 2 as dataset is undirected
    with tqdm(total=int(len(set_pos_edge_index))) as pbar:
        while len(edge_list) < int(len(set_pos_edge_index)):

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
                            # add undirected edges
                            edge_list.append([source_node_id,target_node_id]) 
                            edge_list.append([target_node_id,source_node_id]) 
                            pbar.update(2)

                    else:
                        # add undirected edges
                        edge_list.append([source_node_id,target_node_id]) 
                        edge_list.append([target_node_id,source_node_id]) 
                        pbar.update(2)


    return torch.from_numpy(np.array(edge_list).T)
