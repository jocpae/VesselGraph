
import torch
import numpy as np
import ogb
from ogb.linkproppred import LinkPropPredDataset
import matplotlib.pyplot as plt
from scipy import stats

def plot_nodes(nodes, data, str="train"):

    # generate unique node lists
    u, indices = np.unique(nodes, return_index=True)
    x = []
    y = []
    z = []

    for element in u:
        x.append(float(data['node_feat'][element].flatten()[0]))
        y.append(float(data['node_feat'][element].flatten()[1]))
        z.append(float(data['node_feat'][element].flatten()[2]))

    plt.cla()
    plt.scatter(x,y)
    plt.savefig(f'{str}_xy.png')
    plt.cla()
    plt.scatter(x,z)
    plt.savefig(f'{str}_xz.png')
    plt.cla()
    plt.scatter(y,z)
    plt.savefig(f'{str}_yz.png')


def plot_dist(edges, data, str=""):
    length = []

    for row in edges:
        x1 = data['node_feat'][row[0]][0]
        x2= data['node_feat'][row[1]][0]
        y1 = data['node_feat'][row[0]][1]
        y2 = data['node_feat'][row[1]][1]
        z1 = data['node_feat'][row[0]][2]
        z2= data['node_feat'][row[1]][2]
        length.append(np.sqrt((x2-x1)** 2 + (y2-y1)** 2+ (z2-z1)** 2))

    length = np.array(length)
    q25, q75 = np.percentile(length, [25, 75])
    bin_width = 2 * (q75 - q25) * len(length) ** (-1/3)
    bins = round((length.max() - length.min()) / bin_width)
    plt.cla()
    plt.hist(length, density=True, bins=bins) 
    plt.savefig(f'{str}_vessel_hist.png')
    return np.array(length)


def main():

    dataset = LinkPropPredDataset(name='ogbl-vessel')
    data = dataset[0]

    edge_index = data['edge_index']

    print(f'Number of undirected training edges in the graph (accessing edge_index) :{edge_index.shape[1]}')
    print(f'Number of nodes in the graph:{data["num_nodes"]}')

    print("Examining the splits")

    split = dataset.get_edge_split()

    split_train_pos = split['train']['edge'].cpu().detach().numpy()
    split_train_neg = split['train']['edge_neg'].cpu().detach().numpy()
    split_test_pos = split['test']['edge'].cpu().detach().numpy()
    split_test_neg = split['test']['edge_neg'].cpu().detach().numpy()
    split_valid_pos = split['valid']['edge'].cpu().detach().numpy()
    split_valid_neg = split['valid']['edge_neg'].cpu().detach().numpy()

    # print dimensions of all

    print(f'Dimensions of positive training edges: {split_train_pos.shape[0]}')
    print(f'Dimensions of positive test edges {split_test_pos.shape[0]}')
    print(f'Dimensions of positive validation edges {split_valid_pos.shape[0]}')
    print(f'Dimensions of negative training edges {split_train_neg.shape[0]}')
    print(f'Dimensions of negative test edges {split_test_neg.shape[0]}')
    print(f'Dimensions of negative validaztion edges {split_valid_neg.shape[0]}')

    u, indices = np.unique(split_train_pos, return_index=True, axis=0)
    print("Train edge pos - unique elements", len(u))
    u, indices = np.unique(split_train_neg, return_index=True, axis=0)
    print("Train edge neg - unique elements", len(u))

    u, indices = np.unique(split_valid_pos, return_index=True, axis=0)
    print("Valid edge pos - unique elements", len(u))
    u, indices = np.unique(split_valid_neg, return_index=True, axis=0)
    print("Valid edge neg - unique elements", len(u))

    u, indices = np.unique(split_test_pos, return_index=True, axis=0)
    print("Test edge pos - unique elements", len(u))
    u, indices = np.unique(split_test_neg, return_index=True, axis=0)
    print("Test edge neg - unique elements", len(u))

    print("#####################################")

    array = np.concatenate((split_train_pos,split_train_neg,split_test_pos,split_test_neg,split_valid_pos,split_valid_neg),axis=0)
    u, indices = np.unique(array, return_index=True, axis=0)
    print("All unique elements calculated", len(u))
    print(f'All unique elements in theory: {split_train_pos.shape[0]*2 + split_valid_pos.shape[0]*2 + split_test_pos.shape[0]*2}')

    print("Is the data representive - e.g. are test and validation set similar to training set")

    # generate unique node lists

    train_nodes = np.array(split_train_pos).flatten()
    valid_nodes = np.array(split_valid_pos).flatten()
    test_nodes = np.array(split_test_pos).flatten()

    plot_nodes(train_nodes, data, str="train")
    plot_nodes(valid_nodes, data, str="valid")
    plot_nodes(test_nodes, data, str="test")

    # plot distributions of vessels

    print("")

    train_pos_hist = plot_dist(split_train_pos, data)
    valid_pos_hist = plot_dist(split_valid_pos, data)
    test_pos_hist = plot_dist(split_test_pos, data)

    stat, pvalue = stats.ks_2samp(train_pos_hist,valid_pos_hist) 
    print(f'KS-Test stats, p-value train vs valid set {stat, pvalue}')

    stat, pvalue = stats.ks_2samp(train_pos_hist,test_pos_hist) 
    print(f'KS-Test stats, p-value test vs valid set {stat, pvalue}')

    stat, pvalue = stats.ks_2samp(valid_pos_hist,test_pos_hist) 
    print(f'KS-Test stats, p-value valid vs test set {stat, pvalue}')

    # negative edges

    train_neg_hist = plot_dist(split_train_neg, data)
    valid_neg_hist = plot_dist(split_valid_neg, data)
    test_neg_hist = plot_dist(split_test_neg, data)

    stat, pvalue = stats.ks_2samp(train_neg_hist,valid_neg_hist) 
    print(f'KS-Test stats, p-value train neg vs valid neg set {stat, pvalue}')

    stat, pvalue = stats.ks_2samp(train_neg_hist,test_neg_hist) 
    print(f'KS-Test stats, p-value train neg vs test neg set {stat, pvalue}')

    stat, pvalue = stats.ks_2samp(valid_neg_hist,test_neg_hist) 
    print(f'KS-Test stats, p-value valid neg vs test neg set {stat, pvalue}')




if __name__ == "__main__":
    main()
