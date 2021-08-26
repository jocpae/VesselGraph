import pandas as pd
import numpy as np
import argparse
import os
import argparse
import os
import os.path as osp
import matplotlib.pyplot as plt

print("calculating the branching angles!!")

# cf.: https://arxiv.org/pdf/1803.09340.pdf
# cos(phi_l) = rp^⁴ 

# cf. https://stackoverflow.com/questions/62888678/saving-a-3d-graph-generated-in-networkx-to-vtk-format-for-viewing-in-paraview

def get_values_as_tuple(dict_list, keys):
        return [tuple(d[k] for k in keys) for d in dict_list]   

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

'''
parser = argparse.ArgumentParser(description='calculates branching angles of voreen csv based graph.')
parser.add_argument('-n','--node_list', help='Name of the node list csv file.', required=True)
parser.add_argument('-e','--edge_list', help='Filtering condition: average radius of the vessel.', required=True)
parser.add_argument('-o','--output_directory', help='Output name of branching angles.', type=dir_path, required=True)

# holds all graphs
data_list = []

# read the arguments
args = vars(parser.parse_args())

node_path = args['node_list']
edge_path = args['edge_list']
output_path = os.path.abspath(args['output_directory'])

# 1) nodes

print("Rendering Nodes")

df_nodes = pd.read_csv(node_path,sep=';')

# debugging
df_nodes.info(verbose=True)

# Node feature matrix with shape [num_nodes, num_node_features]
x = np.array(df_nodes[['degree','isAtSampleBorder']].to_numpy())

# data.pos: Node position matrix with shape [num_nodes, num_dimensions]
pos = np.array( df_nodes[['pos_x', 'pos_y', 'pos_z']].to_numpy())
pos = np.abs(pos)

print(nodes)

# https://en.wikipedia.org/wiki/Spherical_coordinate_system

# r = sqrt(x^2 + y^2 + z^2)
r = np.sqrt(np.square(pos[0,:]) + np.square(pos[1,:]) +np.square(pos[2,:]))
# lambda = arc cos (y/z)
theta = np.arccos(pos[2,:] / r)
# phi = arctan (z /sqrt(x²+y²))
phi = np.arctan(np.divide(pos[1,:] / pos[0,:])

print(r.shape)
print(theta.shape)
print(theta.shape)

#nodes = vd.Points(pos, r=20).c('red')
#node_path = os.path.join(output_path , os.path.splitext(os.path.basename(node_path))[0] + '.vtk')
#vd.write(nodes, node_path)

# 2) edges:

# remove memory
df_nodes = []
x = []
nodes = []

df_edges = pd.read_csv(edge_path,sep=';')
df_edges.info(verbose=True)

# Graph connectivity in COO format with shape [2, num_edges] and type torch.long
edge_index = np.array(df_edges[['node1id','node2id']].to_numpy())

# Edge feature  matrix with shape[num_edges, num_edge_features]
#edge_attr_keys = ['length','distance','curveness','avgCrossSection'] # any many more
edge_attr_keys = ['length'] # any many more
edge_attr = np.array(df_edges[edge_attr_keys].to_numpy())

edges = []

for i in range (0, len(edge_index)):

     edge_1 = edge_index[i,0]
     edge_2 = edge_index[i,1]
     coord_1 = pos[edge_1]
     coord_2 = pos[edge_2]
     edges.append([coord_1, coord_2])

#edges = vd.Lines(edges).lw(20).c('green')
#points = Points(points, r=20).c('black')

#edge_path = os.path.join(output_path , os.path.splitext(os.path.basename(edge_path))[0] + '.vtk')
#vd.write(edges, edge_path)
'''

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

parser = argparse.ArgumentParser(description='generate vectors')
parser.add_argument('-i','--input', help='Name of input directory', required=True)
#parser.add_argument('-o','--output_directory', help='Output name of the filtered csv file.', type=dir_path, required=True)

# read the arguments
args = vars(parser.parse_args())

folder = args['input']
#output_path = os.path.abspath(args['output_directory'])
#print("Path")

subfolders = [ f.path for f in os.scandir(folder) if f.is_dir() ]
print(subfolders)

for i in range(len(subfolders)):

    # get the identifier
    id = os.path.basename(os.path.normpath(subfolders[i]))

    # read csv files for nodes and edges
    #df_nodes = pd.read_csv(osp.join(self.raw_dir, self.name,f'{i}_nodes.csv'),sep=';')
    #df_edges = pd.read_csv(osp.join(self.raw_dir, self.name,f'{i}_edges.csv'),sep=';')

    #df_nodes = pd.read_csv(osp.join(self.raw_dir, self.name,f'{i}_nodes.csv'),sep=';')
    #df_edges = pd.read_csv(osp.join(self.raw_dir, self.name,f'{i}_edges.csv'),sep=';')

    print("Processing:")

    print(osp.join(folder,id, f'{id}_nodes.csv'))
    print(osp.join(folder,id, f'{id}_edges.csv'))

    df_nodes = pd.read_csv(osp.join(folder,id, f'{id}_nodes.csv'),sep=';')
    df_edges = pd.read_csv(osp.join(folder,id, f'{id}_edges.csv'),sep=';')

    #print(df_nodes.head)
    #print(df_edges.head)

    # debugging
    df_nodes.info(verbose=True)

    # Node feature matrix with shape [num_nodes, num_node_features]
    x = np.array(df_nodes[['degree','isAtSampleBorder']].to_numpy())

    # data.pos: Node position matrix with shape [num_nodes, num_dimensions]
    pos = np.array( df_nodes[['pos_x', 'pos_y', 'pos_z']].to_numpy())
    pos = np.abs(pos)

    print("Pos", pos.shape)

    # https://en.wikipedia.org/wiki/Spherical_coordinate_system

    pos = pos.T

    # r = sqrt(x^2 + y^2 + z^2)
    r = np.sqrt(np.square(pos[0,:]) + np.square(pos[1,:]) +np.square(pos[2,:]))
    # lambda = arc cos (y/z)
    theta = np.arccos(pos[2,:] / r)
    # phi = arctan (z /sqrt(x²+y²))
    phi = np.arctan(pos[1,:] / pos[0,:])

    #print(r.shape)
    #print(theta.shape)
    #print(theta.shape)

    edge_attr_r = np.zeros(len(df_edges))
    edge_attr_theta = np.zeros(len(df_edges))
    edge_attr_phi = np.zeros(len(df_edges))

    for i in range(len(df_edges)):
        #print(int(df_edges.iloc[i]['node1id']))
        #print(int(df_edges.iloc[i]['node2id']))
        coord1_r = r[int(df_edges.iloc[i]['node1id'])]
        coord2_r = r[int(df_edges.iloc[i]['node2id'])]
        coord1_theta = theta[int(df_edges.iloc[i]['node1id'])]
        coord2_theta = theta[int(df_edges.iloc[i]['node2id'])]
        coord1_phi = phi[int(df_edges.iloc[i]['node1id'])]
        coord2_phi = phi[int(df_edges.iloc[i]['node2id'])]

        # calculate edge feature
        edge_attr_r[i] = coord2_r - coord1_r
        edge_attr_theta[i] = coord2_theta - coord1_theta
        edge_attr_phi[i] = coord2_phi - coord1_phi

    df_edges = pd.read_csv(osp.join(folder,id, f'{id}_edges.csv'),sep=';')
    df_edges.insert(1,"edge_r", list(edge_attr_r),True)
    df_edges.insert(2,"edge_theta", list(edge_attr_theta),True)
    df_edges.insert(3,"edge_phi", list(edge_attr_phi),True)

    file_name = osp.join(folder,id, f'{id}_edges_extended.csv')
    df_edges.to_csv(file_name, sep=';')






