# vessap open graph benchmark dataset

import os
from ogb.io import DatasetSaver
from ogb.linkproppred import LinkPropPredDataset

# step 1

# Create a constructor of DatasetSaver. dataset_name needs to follow OGB convention 
# and start from either ogbn-, ogbl-, or ogbg-. is_hetero is True for heterogeneous graphs, 
# and version indicates the dataset version.

saver = DatasetSaver(dataset_name = 'ogbl-vessel',
                    is_hetero = False,
                    version = 1)

# step 2:
# Create graph_list, storing your graph objects, and call saver.save_graph_list(graph_list).
# Graph objects are dictionaries containing the following keys.

# load "old" OGB Graph
dataset_old = LinkPropPredDataset("ogbl-vessel")
data = dataset_old[0]  # Get the first graph object.

graph_list = []
num_data = len(dataset_old)

for i in range(len(dataset_old)):
    data = dataset_old[i]
    graph = dict()
    graph['num_nodes'] = data['num_nodes']
    graph['node_feat'] = data['node_feat'][:,0:3] # only X,Y,Z coordinates
    graph['edge_index'] = data['edge_index'] # only train pos edge index, but both directions / undirected!
    graph_list.append(graph)

print(graph_list)
# saving a list of graphs
saver.save_graph_list(graph_list)
# step 4

# Prepare split_idx, a dictionary with three keys, train, valid, and test, and mapping into data indices of numpy.ndarray. Then, call saver.save_split(split_idx, split_name = xxx)

# assign indices

# ogb only stores one directional information for the edges, so we drop every second column

split_edge = dataset_old.get_edge_split()

saver.save_split(split_edge, split_name = 'spatial')

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

saver.save_task_info(task_type = 'link prediction', eval_metric = 'rocauc')

# step 7

meta_dict = saver.get_meta_dict()
print(meta_dict)

# zip and clean up
saver.zip()
saver.cleanup()



