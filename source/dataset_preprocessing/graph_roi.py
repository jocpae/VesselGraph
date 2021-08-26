import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description='converts voreen csv based graph to vtk files .')
parser.add_argument('-n','--node_list', help='Name of the node list csv file.', required=True)
parser.add_argument('-e','--edge_list', help='Filtering condition: average radius of the vessel.', required=True)

parser.add_argument('-x','--x_centre',help='X',type=int)
parser.add_argument('-y','--y_centre',help='Y',type=int)
parser.add_argument('-z','--z_centre',help='Z',type=int)
parser.add_argument('-d','--delta',help="delta of cube",type=int)

args = parser.parse_args()

 # create filter conditions
x_min = args.x_centre - args.delta
x_max = args.x_centre + args.delta
y_min = args.y_centre - args.delta
y_max = args.y_centre + args.delta
z_min = args.z_centre - args.delta
z_max = args.z_centre + args.delta

df_nodes = pd.read_csv(args.node_list, sep=';')
df_edges = pd.read_csv(args.edge_list, sep=';')

df_nodes= df_nodes[(df_nodes['pos_x'] <= x_max) & (df_nodes['pos_x']>= x_min)]
df_nodes= df_nodes[(df_nodes['pos_y'] <= y_max) & (df_nodes['pos_y']>= y_min)]
df_nodes= df_nodes[(df_nodes['pos_z'] <= z_max) & (df_nodes['pos_z']>= z_min)]
nodeids = df_nodes['id'].to_numpy()
# node1id, node2id
edges = df_edges.to_numpy()
left_idx = np.array(df_edges['node1id'],dtype=int)
right_idx = np.array(df_edges['node2id'],dtype=int)
array = []
for i in range(len(df_edges.index)):
	if left_idx[i] in nodeids & right_idx[i] in nodeids:
		array.append(i)

df_edges = df_edges.iloc[array]

edge_path = args.edge_list.split('.csv')[0] + f'_roi_{args.x_centre}_{args.y_centre}_{args.z_centre}.csv'
node_path = args.node_list.split('.csv')[0] + f'_roi_{args.x_centre}_{args.y_centre}_{args.z_centre}.csv'

print(len(df_nodes.index))
print(len(df_edges.index))

df_nodes.reset_index(drop=True, inplace=True)
# reset the index and edge id!
df_edges.reset_index(drop=True, inplace=True)
df_edges['id'] = np.arange(0,len(df_edges.index),1)

df_edges.to_csv(edge_path,index=False,sep=';')
df_nodes.to_csv(node_path,index=False,sep=';')





