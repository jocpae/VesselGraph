import os
import os.path as osp
import torch
import random
import numpy as np
from tqdm import tqdm
import pandas as pd
from glob import glob

from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_gz, extract_tar, extract_zip)
from torch_geometric.data.makedirs import makedirs
from torch_geometric.utils import to_undirected, remove_isolated_nodes, remove_self_loops
from torch_sparse import coalesce

from vessap_utils import *

class LinkVesselGraph(InMemoryDataset):
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

        'BALBc_no1': {'folder': 'BALBc_no1.zip',
                      'url': 'https://syncandshare.lrz.de/dl/fiG21AiiCJE6mVRo6tUsNp4N/BALBc_no1.zip',
                      'AlanBrainAtlas': True},
        'BALBc_no2': {'folder': 'BALBc-no2.zip',
                      'url': 'https://syncandshare.lrz.de/dl/fiS6KM5NvGKfLFrjiCzQh1X1/BALBc_no2.zip',
                      'AlanBrainAtlas': True},
        'BALBc_no3': {'folder': 'BALBc-no3.zip',
                      'url': 'https://syncandshare.lrz.de/dl/fiD9e98baTK3FWC9iPhLQWd8/BALBc_no3.zip',
                      'AlanBrainAtlas': True},
        'C57BL_6_no1': {'folder': 'C57BL_6_no1.zip',
                     'url': 'https://syncandshare.lrz.de/dl/fiVTuLxJeLrqyWdMBy5BGrug/C57BL_6_no1.zip',
                     'AlanBrainAtlas': True},
        'C57BL_6_no2': {'folder': 'C57BL_6_no2.zip',
                     'url': 'https://syncandshare.lrz.de/dl/fiNFpZd5S9NYvUYzNwLgf5gW/C57BL_6_no2.zip',
                     'AlanBrainAtlas': True},
        'C57BL_6_no3': {'folder': 'C57BL_6_no3.zip',
                     'url': 'https://syncandshare.lrz.de/dl/fi3Z62oab67735GLQXZyd2Wd/C57BL_6_no3.zip',
                     'AlanBrainAtlas': True},
        'CD1-E_no1': {'folder': 'CD1-E-no1.zip',
                      'url': 'https://syncandshare.lrz.de/dl/fiQs4v6kXvGBviqnuT7BAxjK/CD1-E_no1.zip',
                      'AlanBrainAtlas': True},
        'CD1-E_no2': {'folder': 'CD1-E-no2.zip',
                      'url': 'https://syncandshare.lrz.de/dl/fiJf6ukkGCdUQwXBKd4Leusp/CD1-E_no2.zip',
                      'AlanBrainAtlas': True},
        'CD1-E_no3': {'folder': 'CD1-E-no3.zip',
                      'url': 'https://syncandshare.lrz.de/dl/fiBkjGNxm7XW5R4gFTWp5MFP/CD1-E_no3.zip',
                      'AlanBrainAtlas': True},

        ## selected regions of interest
        'link_vessap_roi1':{'folder': 'link_vessap_roi1.zip',
            'url': 'https://syncandshare.lrz.de/dl/fiWes5GoXWV1AJNVGWbK34cr/link_vessap_roi1.zip',
            'AlanBrainAtlas': False},
        'link_vessap_roi3': {'folder': 'link_vessap_roi3.zip',
            'url': 'https://syncandshare.lrz.de/dl/fiKNpy5GZTwzfYjHnAJ1QgLP/link_vessap_roi3.zip',
            'AlanBrainAtlas': False},
    }
 
    def __init__(self, root, name, splitting_strategy='spatial',
                number_of_workers = 8,
                val_ratio = 0.1, test_ratio = 0.1,
                use_edge_attr: bool = True,
                use_atlas: bool = False,
                transform=None, pre_transform=None):
 
        self.name = name

        print("Available Datasets are:", self.available_datasets.keys())

        # check if dataset name is valid
        assert self.name in self.available_datasets.keys()

        self.url = self.available_datasets[self.name]['url']
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.use_edge_attr = use_edge_attr
        self.use_atlas = use_atlas
        self.splitting_strategy = splitting_strategy
        self.number_of_workers = int(number_of_workers)

        self.AlanBrainAtlas =  self.available_datasets[self.name]['AlanBrainAtlas']       

        super(LinkVesselGraph, self).__init__(root, transform, pre_transform)
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
        folder = osp.join(self.raw_dir, self.available_datasets[self.name]['folder'])  
        subfolders = sorted([ f.path for f in os.scandir(folder) if f.is_dir() ])
        raw_file_names = [] 
        for i in range(len(subfolders)):
            # get the identifier
            id = os.path.basename(os.path.normpath(subfolders[i]))
            raw_file_names.add(osp.join(self.raw_dir, self.name, id, f'{id}_nodes_processed.csv'))
            raw_file_names.add(osp.join(self.raw_dir, self.name, id, f'{id}_edges_processed.csv'))
            if self.AlanBrainAtlas:
                raw_file_names.add(osp.join(self.raw_dir, self.name, id, f'{id}_atlas_processed.csv'))

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

        data_list = []
        # get subfolders of each mouse brain
        folder = osp.join(self.raw_dir, self.name)
        subfolders = sorted([f.path for f in os.scandir(folder) if f.is_dir()])

        for i in range(len(subfolders)):

            id = os.path.basename(os.path.normpath(subfolders[i]))
            print(osp.join(self.raw_dir, self.name, id, f'{id}_nodes_processed.csv'))
            print(osp.join(self.raw_dir, self.name, id, f'{id}_edges_processed.csv'))

            df_nodes = pd.read_csv(osp.join(self.raw_dir, self.name, id,f'{id}_nodes_processed.csv'),sep=';')
            df_edges = pd.read_csv(osp.join(self.raw_dir, self.name, id,f'{id}_edges_processed.csv'),sep=';')
            data = Data()

            # merge nodes and one hot encoded atlas labels
            if self.AlanBrainAtlas and self.use_atlas:

                df_atlas = pd.read_csv(osp.join(self.raw_dir, self.name, id,f'{id}_atlas_processed.csv'),sep=';')
                df_nodes = df_nodes.join(df_atlas)
                data.node_attr_keys = ['pos_x','pos_y','pos_z','degree','isAtSampleBorder'] + list(df_atlas.columns.values)

            else:
                data.node_attr_keys = ['pos_x','pos_y','pos_z','degree','isAtSampleBorder']

            # Node feature matrix with shape [num_nodes, num_node_features]
            data.x = torch.from_numpy(np.array(df_nodes[data.node_attr_keys].to_numpy()))
           
            # Node position matrix with shape [num_nodes, num_dimensions]
            data.pos = torch.from_numpy(np.array( df_nodes[['pos_x', 'pos_y', 'pos_z']].to_numpy())) # coordinates

            edges = np.column_stack((np.array(df_edges[['node1id']]),np.array(df_edges[['node2id']])))
            data.edge_attr_keys = ['length','distance','avgRadiusAvg','roundnessAvg','curveness']
            edge_features = np.array(df_edges[data.edge_attr_keys].to_numpy())

            data.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            data.edge_attr = torch.from_numpy(np.array(edge_features))

            # remove self loops
            data.edge_index , data.edge_attr = remove_self_loops(data.edge_index,data.edge_attr)

            # filter out isolated nodes
            data.edge_index, data.edge_attr , node_mask = remove_isolated_nodes(edge_index=data.edge_index,edge_attr = data.edge_attr,num_nodes=data.num_nodes)
            data.x = data.x[node_mask]
            data.pos = data.pos[node_mask]

            # problem does not order them as I would like
            # data.edge_index, data.edge_attr = to_undirected(edge_index=data.edge_index,edge_attr = data.edge_attr,num_nodes=data.num_nodes,reduce="add") # add attribute

            edge_array = np.ones((2,int(2*data.edge_attr.shape[0])))
            edge_attr_array = np.ones((2*data.edge_attr.shape[0],data.edge_attr.shape[1]))

            for i in range(0,data.edge_attr.shape[0]):
                edge_array[0,2*i] = int(data.edge_index[0,i])
                edge_array[1, 2*i] = int(data.edge_index[1, i])
                edge_array[1, 2*i+1] = int(data.edge_index[0,i])
                edge_array[0, 2*i+1] = int(data.edge_index[1, i])
                edge_attr_array[2*i,:] = np.array(data.edge_attr[i,:])
                edge_attr_array[2*i+1,:] = np.array(data.edge_attr[i,:])

            # includes all edges (train+test+val) in both drections
            data.edge_index_undirected = torch.tensor(edge_array, dtype=torch.long)
            data.edge_attr_undirected = torch.tensor(edge_attr_array, dtype=torch.long)
            del edge_array
            del edge_attr_array

            if self.splitting_strategy == 'spatial': # sample negative edges only in spatial proximity of nodes (mean + 2sigma)

                data = positive_train_test_split_edges(data, val_ratio=self.val_ratio, test_ratio = self.test_ratio)

                # 100 % in node surroundings

                n_train = data.train_pos_edge_index.shape[1]
                n_test = data.test_pos_edge_index.shape[1]
                n_val = data.val_pos_edge_index.shape[1]

                data = negative_sampling(data.edge_index_undirected,df_edges, data, data.edge_index_undirected, n_train, n_test, n_val,self.number_of_workers)


            elif self.splitting_strategy == 'random': # only randomly sampled edges

                data = custom_train_test_split_edges(data, val_ratio=self.val_ratio, test_ratio = self.test_ratio)

            else:
                raise ValueError('Splitting strategy unknown!')

            if self.use_edge_attr == False:
                del data.train_pos_edge_attr
                del data.test_pos_edge_attr
                del data.val_pos_edge_attr
                del data.edge_attr
                del data.edge_attr_keys
                
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



