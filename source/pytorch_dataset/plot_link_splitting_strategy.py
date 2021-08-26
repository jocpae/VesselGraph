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

# ensure you are not allocating wrong CUDA devices
args = parser.parse_args()
selected_dataset = args.dataset

# import PyTorch libs
from link_dataset import LinkVesselGraph

import seaborn as sns
import matplotlib.pyplot as plt

def main():

    # DISPLAY distribution 

    # 100 % RANDOM SPLIT

    dataset = LinkVesselGraph(root='plotting/data/random', name=selected_dataset, splitting_strategy='random', use_edge_attr=True)
    data = dataset[0]  # Get the first graph object.

    keys = list(data.edge_attr_keys)
    all_edge_attr = np.array(data.edge_attr_undirected)
    train_pos_attr_random = np.array(data.edge_attr)
    test_pos_attr_random = np.array(data.test_pos_edge_attr)
    val_pos_attr_random = np.array(data.val_pos_edge_attr)

    # plot the distplot of all edges

    fig1, axs1 = plt.subplots(ncols=1)
    sns.histplot(all_edge_attr[:, 1],ax=axs1)
    fig1.savefig('random1.png')
    fig1.savefig('random1.pdf')

    fig2, axs2 = plt.subplots(ncols=3)
    fig2.suptitle('Distance Distribution of positive edges in random splits', fontsize=14)
    sns.histplot(all_edge_attr[:, 1], ax=axs2[0])
    sns.histplot(test_pos_attr_random[:, 1], ax=axs2[1])
    sns.histplot(val_pos_attr_random[:, 1], ax=axs2[2])

    plt.setp(axs2, xlabel='distance')
    plt.setp(axs2, ylabel='count')

    axs2[0].set_title(f'train_pos')
    axs2[1].set_title(f'test_pos')
    axs2[2].set_title(f'val_pos')

    plt.tight_layout()
    plt.show()
    fig2.savefig('random2.png',dpi=500)
    fig2.savefig('random2.pdf',dpi=500)

    # negative edges!
    # we have to compute this by ourself!

    keys = list(data.edge_attr_keys)
    all_edge_attr = np.array(data.edge_attr_undirected)

    train_neg_attr_random = []
    test_neg_attr_random = []
    val_neg_attr_random = []

    test_neg_edge_index = np.array(data.test_neg_edge_index)
    train_neg_edge_index = np.array(data.train_neg_edge_index)
    val_neg_edge_index = np.array(data.val_neg_edge_index)

    for i in range(0,test_neg_edge_index.shape[1]):
        x1, y1, z1 = np.array(data.pos[test_neg_edge_index[0,i]])
        x2, y2, z2 = np.array(data.pos[test_neg_edge_index[1,i]])
        test_neg_attr_random.append(np.sqrt(np.power((x1-x2),2)+np.power((y1-y2),2)+np.power((z1-z2),2)))

    for i in range(0,train_neg_edge_index.shape[1]):
        x1, y1, z1 = np.array(data.pos[train_neg_edge_index[0,i]])
        x2, y2, z2 = np.array(data.pos[train_neg_edge_index[1,i]])
        train_neg_attr_random.append(np.sqrt(np.power((x1-x2),2)+np.power((y1-y2),2)+np.power((z1-z2),2)))

    for i in range(0,val_neg_edge_index.shape[1]):
        x1, y1, z1 = np.array(data.pos[val_neg_edge_index[0,i]])
        x2, y2, z2 = np.array(data.pos[val_neg_edge_index[1,i]])
        val_neg_attr_random.append(np.sqrt(np.power((x1-x2),2)+np.power((y1-y2),2)+np.power((z1-z2),2)))

    # plot the distplot of all negative edges

    fig3, axs3 = plt.subplots(ncols=3)
    fig3.suptitle('Distance Distribution of negative edges in 100% random splits', fontsize=14)
    sns.histplot(train_neg_attr_random, ax=axs3[0])
    sns.histplot(test_neg_attr_random, ax=axs3[1])
    sns.histplot(val_neg_attr_random, ax=axs3[2])

    plt.setp(axs3, xlabel='distance')
    plt.setp(axs3, ylabel='count')

    axs3[0].set_title(f'train_neg')
    axs3[1].set_title(f'test_neg')
    axs3[2].set_title(f'val_neg')

    plt.tight_layout()
    plt.show()
    fig3.savefig('random3.png', dpi=500)
    fig3.savefig('random3.pdf', dpi=500)

    print(np.mean(train_neg_attr_random))
    print(np.mean(val_neg_attr_random))
    print(np.mean(test_neg_attr_random))


    ####################################################################################################
    ####################################################################################################
    ####################################################################################################

    ### 100 % SPATIAL

    # 100 % SPATIAL

    dataset_split = LinkVesselGraph(root='plotting/data/spatial', name=selected_dataset,
                                    splitting_strategy='spatial',
                                    use_edge_attr=True)
    data_split = dataset_split[0]  # Get the first graph object.

    all_edge_attr = np.array(data_split.edge_attr_undirected)
    train_pos_attr_spatial = np.array(data_split.edge_attr)
    test_pos_attr_spatial = np.array(data_split.test_pos_edge_attr)
    val_pos_attr_spatial = np.array(data_split.val_pos_edge_attr)

    # plot the distplot of all edges

    fig3, axs3 = plt.subplots(ncols=1)
    sns.histplot(all_edge_attr[:, 1], ax=axs3)
    fig3.savefig('spatial1.png')
    fig3.savefig('spatial1.pdf')

    fig4, axs4 = plt.subplots(ncols=3)
    fig4.suptitle('Distance Distribution of positive edges in spatial splits', fontsize=14)
    sns.histplot(all_edge_attr[:, 1], ax=axs4[0])
    sns.histplot(test_pos_attr_spatial[:, 1], ax=axs4[1])
    sns.histplot(val_pos_attr_spatial[:, 1], ax=axs4[2])

    plt.setp(axs4, xlabel='distance')
    plt.setp(axs4, ylabel='count')

    axs4[0].set_title(f'train_pos')
    axs4[1].set_title(f'test_pos')
    axs4[2].set_title(f'val_pos')

    plt.tight_layout()
    plt.show()
    fig4.savefig('spatial2.png', dpi=500)
    fig4.savefig('spatial2.pdf', dpi=500)

    # negative edges!
    # we have to compute this by ourself!

    keys = list(data_split.edge_attr_keys)
    all_edge_attr = np.array(data_split.edge_attr_undirected)

    train_neg_attr_spatial = []
    test_neg_attr_spatial = []
    val_neg_attr_spatial = []

    test_neg_edge_index = np.array(data_split.test_neg_edge_index)
    train_neg_edge_index = np.array(data_split.train_neg_edge_index)
    val_neg_edge_index = np.array(data_split.val_neg_edge_index)

    for i in range(0, test_neg_edge_index.shape[1]):
        x1, y1, z1 = np.array(data_split.pos[test_neg_edge_index[0,i]])
        x2, y2, z2 = np.array(data_split.pos[test_neg_edge_index[1,i]])
        test_neg_attr_spatial.append(np.sqrt(np.power((x1 - x2), 2) + np.power((y1 - y2), 2) + np.power((z1 - z2), 2)))

    for i in range(0, train_neg_edge_index.shape[1]):
        x1, y1, z1 = np.array(data_split.pos[train_neg_edge_index[0,i]])
        x2, y2, z2 = np.array(data_split.pos[train_neg_edge_index[1,i]])
        train_neg_attr_spatial.append(np.sqrt(np.power((x1 - x2), 2) + np.power((y1 - y2), 2) + np.power((z1 - z2), 2)))

    for i in range(0, val_neg_edge_index.shape[1]):
        x1, y1, z1 = np.array(data_split.pos[val_neg_edge_index[0,i]])
        x2, y2, z2 = np.array(data_split.pos[val_neg_edge_index[1,i]])
        val_neg_attr_spatial.append(np.sqrt(np.power((x1 - x2), 2) + np.power((y1 - y2), 2) + np.power((z1 - z2), 2)))

    print(np.mean(train_neg_attr_spatial))
    print(np.mean(val_neg_attr_spatial))
    print(np.mean(test_neg_attr_spatial))

    # plot the distplot of all edges

    fig3, axs3 = plt.subplots(ncols=3)
    fig3.suptitle('Distance Distribution of negative edges in 100% spatial splits', fontsize=14)
    sns.histplot(train_neg_attr_spatial, ax=axs3[0])
    sns.histplot(test_neg_attr_spatial, ax=axs3[1])
    sns.histplot(val_neg_attr_spatial, ax=axs3[2])

    plt.setp(axs3, xlabel='distance')
    plt.setp(axs3, ylabel='count')

    axs3[0].set_title(f'train_neg')
    axs3[1].set_title(f'test_neg')
    axs3[2].set_title(f'val_neg')

    plt.tight_layout()
    plt.show()
    fig3.savefig('spatial3.png', dpi=500)
    fig3.savefig('spatial3.pdf', dpi=500)

    ####################################################################################################
    ####################################################################################################

    ### combination: 50% random, 50% spatial

    dataset_split = LinkVesselGraph(root='plotting/data/combination', name=selected_dataset,
                                    splitting_strategy='combination',
                                    use_edge_attr=True)
    data_split = dataset_split[0]  # Get the first graph object.

    all_edge_attr = np.array(data_split.edge_attr_undirected)
    train_pos_attr_combination = np.array(data_split.edge_attr)
    test_pos_attr_combination = np.array(data_split.test_pos_edge_attr)
    val_pos_attr_combination = np.array(data_split.val_pos_edge_attr)

    # plot the distplot of all edges

    fig3, axs3 = plt.subplots(ncols=1)
    sns.histplot(all_edge_attr[:, 1], ax=axs3)
    fig3.savefig('combination1.png')
    fig3.savefig('combination1.pdf')

    fig4, axs4 = plt.subplots(ncols=3)
    fig4.suptitle('Distance Distribution of positive edges in combination splits', fontsize=14)
    sns.histplot(all_edge_attr[:, 1], ax=axs4[0])
    sns.histplot(test_pos_attr_combination[:, 1], ax=axs4[1])
    sns.histplot(val_pos_attr_combination[:, 1], ax=axs4[2])

    plt.setp(axs4, xlabel='distance')
    plt.setp(axs4, ylabel='count')

    axs4[0].set_title(f'train_pos')
    axs4[1].set_title(f'test_pos')
    axs4[2].set_title(f'val_pos')

    plt.tight_layout()
    plt.show()
    fig4.savefig('combination2.png', dpi=500)
    fig4.savefig('combination2.pdf', dpi=500)

    # negative edges!
    # we have to compute this by ourself!

    keys = list(data_split.edge_attr_keys)
    all_edge_attr = np.array(data_split.edge_attr_undirected)

    train_neg_attr_combination = []
    test_neg_attr_combination = []
    val_neg_attr_combination = []

    test_neg_edge_index = np.array(data_split.test_neg_edge_index)
    train_neg_edge_index = np.array(data_split.train_neg_edge_index)
    val_neg_edge_index = np.array(data_split.val_neg_edge_index)

    for i in range(0, test_neg_edge_index.shape[1]):
        x1, y1, z1 = np.array(data_split.pos[test_neg_edge_index[0, i]])
        x2, y2, z2 = np.array(data_split.pos[test_neg_edge_index[1, i]])
        test_neg_attr_combination.append(
            np.sqrt(np.power((x1 - x2), 2) + np.power((y1 - y2), 2) + np.power((z1 - z2), 2)))

    for i in range(0, train_neg_edge_index.shape[1]):
        x1, y1, z1 = np.array(data_split.pos[train_neg_edge_index[0, i]])
        x2, y2, z2 = np.array(data_split.pos[train_neg_edge_index[1, i]])
        train_neg_attr_combination.append(
            np.sqrt(np.power((x1 - x2), 2) + np.power((y1 - y2), 2) + np.power((z1 - z2), 2)))

    for i in range(0, val_neg_edge_index.shape[1]):
        x1, y1, z1 = np.array(data_split.pos[val_neg_edge_index[0, i]])
        x2, y2, z2 = np.array(data_split.pos[val_neg_edge_index[1, i]])
        val_neg_attr_combination.append(
            np.sqrt(np.power((x1 - x2), 2) + np.power((y1 - y2), 2) + np.power((z1 - z2), 2)))

    print(np.mean(train_neg_attr_combination))
    print(np.mean(val_neg_attr_combination))
    print(np.mean(test_neg_attr_combination))

    # plot the distplot of all edges

    fig3, axs3 = plt.subplots(ncols=3)
    fig3.suptitle('Distance Distribution of negative edges in combination splits', fontsize=14)
    sns.histplot(train_neg_attr_combination, ax=axs3[0])
    sns.histplot(test_neg_attr_combination, ax=axs3[1])
    sns.histplot(val_neg_attr_combination, ax=axs3[2])

    plt.setp(axs3, xlabel='distance')
    plt.setp(axs3, ylabel='count')

    axs3[0].set_title(f'train_neg')
    axs3[1].set_title(f'test_neg')
    axs3[2].set_title(f'val_neg')

    plt.tight_layout()
    plt.show()
    fig3.savefig('combination3.png', dpi=500)
    fig3.savefig('combination3.pdf', dpi=500)


if __name__ == "__main__":
        main()



