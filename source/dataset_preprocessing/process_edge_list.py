import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Preprocess edge list')
parser.add_argument('-e','--edge_list', help='Edge List to be processed.', required=True)
parser.add_argument('-n','--node_list', help='Node List to be processed.', required=True)
parser.add_argument('-d','--display_fit',help='Whether to display the fit or not.',action='store_true')
args = parser.parse_args()

# read unfiltered edge list
df_edges = pd.read_csv(args.edge_list,sep=';')

# obtain copy of the node1id and node2ids
edges = np.column_stack((np.array(df_edges[['node1id']]),np.array(df_edges[['node2id']])))

# 0, 6500, 6701
# 1, 4500, 3001
# is converted to
# 0, 6500, 6701
# 1, 3001, 4500

# left index small nodeid, right index bigger nodeid
df_edges.iloc[:,1] = np.array(np.min(edges,axis=1))
df_edges.iloc[:,2] = np.array(np.max(edges,axis=1))

# sort pandas dataframe

# sort the pandas dataframe in ascending order by node1id
df_edges = df_edges.sort_values('node1id')
# rename the index
df_edges.reset_index(drop=True, inplace=True)
df_edges['id'] = np.arange(0,edges.shape[0],1)

# get newly ordered edges
edges = np.column_stack((np.array(df_edges[['node1id']]),np.array(df_edges[['node2id']])))

# obtain the indexes of the first occuring objects
_, indices = np.unique(edges[:,[0,1]],return_index=True,axis=0)

first_occurrence = np.sort(indices)
all_indices = np.arange(0,edges.shape[0],1)
duplicates = np.setdiff1d(all_indices,indices)

# get count of node1id,node2id combination
cnts = df_edges.groupby(['node1id','node2id']).size().reset_index().rename(columns={0:'count'})
filt_cnts = cnts[cnts['count'] >1] # drop all edges that only occur once
counter = np.array(filt_cnts['count'])

helper = []
helpindex = 0

for i in range(0,len(counter)):
	helper.append(duplicates[helpindex])
	helpindex += counter[i] - 1


from scipy.optimize import curve_fit

# fitting function - averageRadisToMean

def func(x, a, c):#
	return a * x + c

y = np.array(df_edges['roundnessAvg'])
x = np.abs(np.array(df_edges['avgRadiusAvg']))

popt, pcov = curve_fit(func, x, y)

if args.display_fit:
	# to check fit
	plt.scatter(x,y,s=1)
	plt.scatter(x,func(x,*popt),s=1)
	plt.show()


for i in range(0,len(counter)):

	# get count of number of instances
	cnt = counter[i]
	index = helper[i] - 1 # we take the first element as the index, not the first duplicate


	# variables
	length = 0
	volume = 0
	num_voxels = 0
	hasNodeAtSampleBorder = 0
	curveness = 0
	avgCrossSection = 0
	avgRadiusAvg = 0
	avgRadiusStd = 0
	roundnessAvg = 0
	roundnessStd = 0

	for k in range(0,cnt):
		length += df_edges.iloc[index+k]['length']
		volume += df_edges.iloc[index + k]['volume']
		num_voxels += df_edges.iloc[index + k]['num_voxels']
		curveness += df_edges.iloc[index + k]['curveness']
		avgCrossSection += df_edges.iloc[index + k]['avgCrossSection']
		avgRadiusAvg += df_edges.iloc[index + k]['avgRadiusAvg']
	
	df_edges.at[index, 'length']= length / cnt # mean
	df_edges.at[index, 'volume'] = volume  # add
	df_edges.at[index, 'num_voxels'] = num_voxels # add
	df_edges.at[index, 'curveness']= curveness / cnt # mean
	df_edges.at[index, 'avgCrossSection']= avgCrossSection # add
	df_edges.at[index, 'avgRadiusAvg'] = avgRadiusAvg # add

	# compute the median

	radius_median = np.median(np.array(df_edges['avgRadiusStd'] / df_edges['avgRadiusAvg']))
	roundness_median = np.median(np.array(df_edges['roundnessStd'] / df_edges['roundnessAvg']))

	df_edges.at[index, 'avgRadiusStd'] = avgRadiusAvg * radius_median 
	df_edges.at[index, 'minRadiusAvg'] = avgRadiusAvg
	df_edges.at[index, 'minRadiusStd'] = avgRadiusAvg * radius_median  
	df_edges.at[index, 'maxRadiusAvg'] = avgRadiusAvg
	df_edges.at[index, 'maxRadiusStd'] = avgRadiusAvg * radius_median  
	df_edges.at[index, 'roundnessAvg'] = func(avgRadiusAvg,*popt) # linear mapping
	df_edges.at[index, 'roundnessStd'] = func(avgRadiusAvg,*popt) * roundness_median

	# hasNodeAtBorder - logical or

	for k in range(0,cnt):
		hasNodeAtSampleBorder += df_edges.iloc[index + k]['hasNodeAtSampleBorder']

	if hasNodeAtSampleBorder > 1:
		hasNodeAtSampleBorder = 1

	df_edges.at[index, 'hasNodeAtSampleBorder'] = hasNodeAtSampleBorder



# remove the duplicate we merged earlier
df_edges.drop(df_edges.index[duplicates], inplace=True)

print("Edge Rows:",len(df_edges.index))


# drop all self loops
df_edges.drop(df_edges[df_edges.node1id == df_edges.node2id].index,inplace=True)
print("Edge Rows:",len(df_edges.index))


# reset the index and edge id!
df_edges.reset_index(drop=True, inplace=True)
df_edges['id'] = np.arange(0,len(df_edges.index),1)

# compute node degrees from scratch

x = np.concatenate([np.array(df_edges[['node1id']]).flatten(),np.array(df_edges[['node2id']]).flatten()])
y = np.bincount(x)
ii = np.nonzero(y)[0]
node_cnt = np.vstack((ii,y[ii])).T

df_nodes= pd.read_csv(args.node_list,sep=';')

node_degree = np.zeros((len(df_nodes.index)))
node_degree[node_cnt[:,0]] = node_cnt[:,1] # contains isolated nodes!

# remove nodes that are not in the edge list!
# compute node1id and node2id for edge list
# compute node_degree for node list
df_nodes[['degree']] = node_degree.reshape(len(node_degree),1).astype(dtype=np.int8)

node1 = df_edges[['node1id']].to_numpy()
node2 = df_edges[['node2id']].to_numpy()

df_edges[['node1_degree']] = np.array(node_degree[node1],dtype=np.uint8)
df_edges[['node2_degree']] = np.array(node_degree[node2],dtype=np.uint8)

# make sure all edge attributes are positive

df_edges = df_edges.abs()

edge_path = args.edge_list.split('.csv')[0] + "_processed.csv"
node_path = args.node_list.split('.csv')[0] + "_processed.csv"

df_edges.to_csv(edge_path,index=False,sep=';')
df_nodes.to_csv(node_path,index=False,sep=';')

