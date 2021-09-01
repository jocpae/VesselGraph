import os
import os.path as osp
import torch
import random
import numpy as np
import pandas as pd

from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_gz, extract_tar, extract_zip)
from torch_geometric.data.makedirs import makedirs
from torch_geometric.utils import to_undirected, remove_isolated_nodes, remove_self_loops

class NodeVesselGraph(InMemoryDataset):
    r"""A variety of generated graph datasets including whole mouse brain vasculature graphs from
    `"Machine learning analysis of whole mouse brain vasculature"
    <https://www.nature.com/articles/s41592-020-0792-1>`_  and
    `"Micrometer-resolution reconstruction and analysis of whole mouse brain vasculature 
    by synchrotron-based phase-contrast tomographic microscopy"
    <https://www.biorxiv.org/content/10.1101/2021.03.16.435616v1.full#fn-3>`_ and
    `"Brain microvasculature has a common topology with local differences in geometry that match metabolic load>`_
    <https://www.sciencedirect.com/science/article/abs/pii/S0896627321000805>`_
    paper.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the (partial dataset / collection) (one of :obj:`"synthetic"`,
            :obj:`"vessap"`, :obj:`"vessapcd"`, :obj:`"italo"`)
        splitting_strategy (string): Random or spatial splitting.
            If :obj:`"random"`, random splitting strategy.
            If :obj:`"spatial"`, spatial splitting strategy.
            If :obj:`"combined"`, 50% / 50% random and spatially sampled links.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        use_edge_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous edge attributes (if present).
            (default: :obj:`True`)
    """

    # file structure is dataset_name/folder_of_file/folder_of_file_{nodes,edges}.csv

    available_datasets = {

        'synthetic': {'folder':'synthetic.zip',
                      'url':'https://syncandshare.lrz.de/dl/fiYbEo8Vv1mpHWtT2ShRqB3i/synthetic.zip',
                      'AlanBrainAtlas':False},

        'synthetic_graph_1': {'folder':'synthetic.zip',
                      'url':'https://syncandshare.lrz.de/dl/fiXfSD14pKGM54L5BqZxF8vF/synthetic_graph_1.zip',
                      'AlanBrainAtlas':False},
        'synthetic_graph_2': {'folder':'synthetic.zip',
                      'url':'https://syncandshare.lrz.de/dl/fiEDhbBHmqawVwKaBeWwHgT8/synthetic_graph_2.zip',
                      'AlanBrainAtlas':False},
        'synthetic_graph_3': {'folder':'synthetic.zip',
                      'url':'https://syncandshare.lrz.de/dl/fiPvTKvqhqNtQ8B6UyGfbvGi/synthetic_graph_3.zip',
                      'AlanBrainAtlas':False},
        'synthetic_graph_4': {'folder':'synthetic.zip',
                      'url':'https://syncandshare.lrz.de/dl/fiFq7BVkRZekbBYQSVYX8L6K/synthetic_graph_4.zip',
                      'AlanBrainAtlas':False},
        'synthetic_graph_5': {'folder':'synthetic.zip',
                      'url':'https://syncandshare.lrz.de/dl/fi5dos737XVZxuyqQ5gmUW6p/synthetic_graph_5.zip',
                      'AlanBrainAtlas':False},

        'BALBc_no1': {'folder': 'BALBc_no1.zip',
                      'url': 'https://syncandshare.lrz.de/dl/fiG21AiiCJE6mVRo6tUsNp4N/BALBc_no1.zip',
                      'AlanBrainAtlas': False},
        'BALBc_no2': {'folder': 'BALBc-no2.zip',
                      'url': 'https://syncandshare.lrz.de/dl/fiS6KM5NvGKfLFrjiCzQh1X1/BALBc_no2.zip',
                      'AlanBrainAtlas': False},
        'BALBc_no3': {'folder': 'BALBc-no3.zip',
                      'url': 'https://syncandshare.lrz.de/dl/fiD9e98baTK3FWC9iPhLQWd8/BALBc_no3.zip',
                      'AlanBrainAtlas': False},
        'C57BL_6_no1': {'folder': 'C57BL_6_no1.zip',
                     'url': 'https://syncandshare.lrz.de/dl/fiVTuLxJeLrqyWdMBy5BGrug/C57BL_6_no1.zip',
                     'AlanBrainAtlas': False},
        'C57BL_6_no2': {'folder': 'C57BL_6_no2.zip',
                     'url': 'https://syncandshare.lrz.de/dl/fiNFpZd5S9NYvUYzNwLgf5gW/C57BL_6_no2.zip',
                     'AlanBrainAtlas': False},
        'C57BL_6_no3': {'folder': 'C57BL_6_no3.zip',
                     'url': 'https://syncandshare.lrz.de/dl/fi3Z62oab67735GLQXZyd2Wd/C57BL_6_no3.zip',
                     'AlanBrainAtlas': False},
        'CD1-E_no1': {'folder': 'CD1-E-no1.zip',
                      'url': 'https://syncandshare.lrz.de/dl/fiQs4v6kXvGBviqnuT7BAxjK/CD1-E_no1.zip',
                      'AlanBrainAtlas': False},
        'CD1-E_no2': {'folder': 'CD1-E-no2.zip',
                      'url': 'https://syncandshare.lrz.de/dl/fiJf6ukkGCdUQwXBKd4Leusp/CD1-E_no2.zip',
                      'AlanBrainAtlas': False},
        'CD1-E_no3': {'folder': 'CD1-E-no3.zip',
                      'url': 'https://syncandshare.lrz.de/dl/fiBkjGNxm7XW5R4gFTWp5MFP/CD1-E_no3.zip',
                      'AlanBrainAtlas': False},

        ## selected regions of interest
        'node_vessap_roi1':{'folder': 'node_vessap_roi1.zip',
            'url': 'https://syncandshare.lrz.de/dl/fi8w9EY1crCyP5aQ7nVpmWKF/node_vessap_roi1.zip',
            'AlanBrainAtlas': False},
        'node_vessap_roi3': {'folder': 'node_vessap_roi3.zip',
            'url': 'https://syncandshare.lrz.de/dl/fiP4SFHzcU6Qkdm9Mbi16pQg/node_vessap_roi3.zip',
            'AlanBrainAtlas': False},
    }

    def __init__(self, root, name, transform=None, pre_transform=None,
                 use_node_attr: bool = True, use_edge_attr: bool = True):

        self.name = name#.lower()

        # check if dataset name is valid
        assert self.name in self.available_datasets.keys()

        self.url = self.available_datasets[self.name]['url']
        self.use_node_attr = use_node_attr
        self.use_edge_attr = use_edge_attr

        super(NodeVesselGraph, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        # get subfolders of each graph
        folder = osp.join(self.raw_dir, self.name)
        subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
        raw_file_names = []
        for i in range(len(subfolders)):
            # get the identifier
            id = os.path.basename(os.path.normpath(subfolders[i]))
            raw_file_names.add(osp.join(self.raw_dir, self.name, id, f'{id}_nodes_processed.csv'))
            raw_file_names.add(osp.join(self.raw_dir, self.name, id, f'{id}_edges_processed.csv'))

        print(raw_file_names)
        return [raw_file_names]

    @property
    def processed_file_names(self):
        return 'dataset.pt'

    def _download(self):
        if osp.isdir(self.raw_dir) and len(os.listdir(self.raw_dir)) > 0:
            return

        makedirs(self.raw_dir)
        self.download()

    def download(self):
        path = download_url(self.url, self.raw_dir, log=True)
        name = self.available_datasets[self.name]['folder']

        if name.endswith('.tar.gz'):
            extract_tar(path, self.raw_dir)
        elif name.endswith('.tar.xz'):
            extract_tar(path, self.raw_dir)
        elif name.endswith('.gz'):
            extract_gz(path, self.raw_dir)
        elif name.endswith('.zip'):
            extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):

        # reproducible results
        np.random.seed(123)
        torch.manual_seed(123)
        np.random.seed(123)


        # holds all graphs
        data_list = []

        # get subfoldes of each mouse brain
        folder = osp.join(self.raw_dir, self.name)
        subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]

        for i in range(len(subfolders)):
            # get the identifier
            id = os.path.basename(os.path.normpath(subfolders[i]))

            # read csv files for nodes and edges

            print(osp.join(self.raw_dir, self.name, id, f'{id}_nodes_processed.csv'))
            print(osp.join(self.raw_dir, self.name, id, f'{id}_edges_processed.csv'))

            df_nodes = pd.read_csv(osp.join(self.raw_dir, self.name, id, f'{id}_nodes_processed.csv'), sep=';')
            df_edges = pd.read_csv(osp.join(self.raw_dir, self.name, id, f'{id}_edges_processed.csv'), sep=';')

            # PyTorch Geometrics Data Class Object
            data = Data()

            # store keys of node and edge features
            data.node_attr_keys = ['pos_x', 'pos_y', 'pos_z', 'degree', 'isAtSampleBorder']
            data.edge_attr_keys = ['length', 'distance', 'curveness', 'volume', 'avgCrossSection',
                                   'minRadiusAvg', 'minRadiusStd', 'avgRadiusAvg', 'avgRadiusStd',
                                   'maxRadiusAvg', 'maxRadiusStd', 'roundnessAvg', 'roundnessStd',
                                   'node1_degree', 'node2_degree', 'num_voxels', 'hasNodeAtSampleBorder']

            # Node feature matrix with shape [num_nodes, num_node_features]
            data.x = torch.from_numpy(np.array(df_nodes[data.node_attr_keys].to_numpy()))

            # Node position matrix with shape [num_nodes, num_dimensions]
            data.pos = torch.from_numpy(np.array(df_nodes[['pos_x', 'pos_y', 'pos_z']].to_numpy()))  # coordinates

            # Graph connectivity COO format with shape [2, num_edges]

            edge_index_source = np.array(df_edges[['node1id']])
            edge_index_sink = np.array(df_edges[['node2id']])
            edges = np.column_stack((edge_index_source, edge_index_sink))

            # Edge feature matrix with shape [num_edges, num_edge_features]
            edge_features = np.array(df_edges[data.edge_attr_keys].to_numpy())

            # Filter vessels

            data.edge_attr = torch.from_numpy(np.array(edge_features))
            data.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

            # convert the graph to an undirected graph
            data.edge_index, data.edge_attr = to_undirected(edge_index=data.edge_index, edge_attr=data.edge_attr,
                                                            num_nodes=data.num_nodes, reduce="add")

            # remove self loops
            data.edge_index, data.edge_attr = remove_self_loops(data.edge_index, data.edge_attr)

            # filter out isolated nodes
            data.edge_index, data.edge_attr, node_mask = remove_isolated_nodes(edge_index=data.edge_index,
                                                                               edge_attr=data.edge_attr,
                                                                               num_nodes=data.num_nodes)
            data.x = data.x[node_mask]
            data.pos = data.pos[node_mask]

            # append to other graphs
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)



