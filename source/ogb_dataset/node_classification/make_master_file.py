### script for writing meta information of datasets into master.csv
### for node property prediction datasets.
import pandas as pd

dataset_dict = {}
dataset_list = []

### add meta-information about protein function prediction task
name = 'ogbn-proteins'
dataset_dict[name] = {'num tasks': 112, 'num classes': 2, 'eval metric': 'rocauc', 'task type': 'binary classification'}
dataset_dict[name]['download_name'] = 'proteins'
dataset_dict[name]['version'] = 1
dataset_dict[name]['url'] = 'http://snap.stanford.edu/ogb/data/nodeproppred/'+dataset_dict[name]['download_name']+'.zip'
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]['add_inverse_edge'] = True 
dataset_dict[name]['has_node_attr'] = False
dataset_dict[name]['has_edge_attr'] = True
dataset_dict[name]['split'] = 'species'
dataset_dict[name]['additional node files'] = 'node_species'
dataset_dict[name]['additional edge files'] = 'None'
dataset_dict[name]['is hetero'] = False
dataset_dict[name]['binary'] = False

### add meta-information about product category prediction task
name = 'ogbn-products'
dataset_dict[name] = {'num tasks': 1, 'num classes': 47, 'eval metric': 'acc', 'task type': 'multiclass classification'}
dataset_dict[name]['download_name'] = 'products'
dataset_dict[name]['version'] = 1
dataset_dict[name]['url'] = 'http://snap.stanford.edu/ogb/data/nodeproppred/'+dataset_dict[name]['download_name']+'.zip'
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]['add_inverse_edge'] = True 
dataset_dict[name]['has_node_attr'] = True
dataset_dict[name]['has_edge_attr'] = False
dataset_dict[name]['split'] = 'sales_ranking'
dataset_dict[name]['additional node files'] = 'None'
dataset_dict[name]['additional edge files'] = 'None'
dataset_dict[name]['is hetero'] = False
dataset_dict[name]['binary'] = False

### add meta-information about arxiv category prediction task
name = 'ogbn-arxiv'
dataset_dict[name] = {'num tasks': 1, 'num classes': 40, 'eval metric': 'acc', 'task type': 'multiclass classification'}
dataset_dict[name]['download_name'] = 'arxiv'
dataset_dict[name]['version'] = 1
dataset_dict[name]['url'] = 'http://snap.stanford.edu/ogb/data/nodeproppred/'+dataset_dict[name]['download_name']+'.zip'
dataset_dict[name]['add_inverse_edge'] = False 
dataset_dict[name]['has_node_attr'] = True
dataset_dict[name]['has_edge_attr'] = False
dataset_dict[name]['split'] = 'time'
dataset_dict[name]['additional node files'] = 'node_year'
dataset_dict[name]['additional edge files'] = 'None'
dataset_dict[name]['is hetero'] = False
dataset_dict[name]['binary'] = False

### add meta-information about paper venue prediction task
name = 'ogbn-mag'
dataset_dict[name] = {'num tasks': 1, 'num classes': 349, 'eval metric': 'acc', 'task type': 'multiclass classification'}
dataset_dict[name]['download_name'] = 'mag'
dataset_dict[name]['version'] = 2
dataset_dict[name]['url'] = 'http://snap.stanford.edu/ogb/data/nodeproppred/'+dataset_dict[name]['download_name']+'.zip'
dataset_dict[name]['add_inverse_edge'] = False 
dataset_dict[name]['has_node_attr'] = True
dataset_dict[name]['has_edge_attr'] = False
dataset_dict[name]['split'] = 'time'
dataset_dict[name]['additional node files'] = 'node_year'
dataset_dict[name]['additional edge files'] = 'edge_reltype'
dataset_dict[name]['is hetero'] = True
dataset_dict[name]['binary'] = False

### add meta-information about paper category prediction in huge paper citation network
name = 'ogbn-papers100M'
dataset_dict[name] = {'num tasks': 1, 'num classes': 172, 'eval metric': 'acc', 'task type': 'multiclass classification'}
dataset_dict[name]['download_name'] = 'papers100M-bin'
dataset_dict[name]['version'] = 1
dataset_dict[name]['url'] = 'http://snap.stanford.edu/ogb/data/nodeproppred/'+dataset_dict[name]['download_name']+'.zip'
dataset_dict[name]['add_inverse_edge'] = False 
dataset_dict[name]['has_node_attr'] = True
dataset_dict[name]['has_edge_attr'] = False
dataset_dict[name]['split'] = 'time'
dataset_dict[name]['additional node files'] = 'node_year'
dataset_dict[name]['additional edge files'] = 'None'
dataset_dict[name]['is hetero'] = False
dataset_dict[name]['binary'] = True

### add meta-information about product category prediction task
name = 'ogbn-BALBc_no1_pb_minRadiusAvg'
dataset_dict[name] = {'num tasks': 1, 'num classes': 3, 'eval metric': 'acc', 'task type': 'multiclass classification'}
dataset_dict[name]['download_name'] = 'BALBc_no1_pb_minRadiusAvg'
dataset_dict[name]['version'] = 1
# usually use the sync and share link; for test, debug and development you can also use a local file link
dataset_dict[name]['url'] = 'https://syncandshare.lrz.de/dl/fi5se6Z9gAXTqH1G3Y1bXmip/BALBc_no1_pb_minRadiusAvg.zip'
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]['add_inverse_edge'] = False
dataset_dict[name]['has_node_attr'] = True
dataset_dict[name]['has_edge_attr'] = True
dataset_dict[name]['split'] = 'random'
dataset_dict[name]['additional node files'] = 'None'
dataset_dict[name]['additional edge files'] = 'None'
dataset_dict[name]['is hetero'] = False
dataset_dict[name]['binary'] = True

### add meta-information about product category prediction task
name = 'ogbn-BALBc_no2_pb_minRadiusAvg'
dataset_dict[name] = {'num tasks': 1, 'num classes': 3, 'eval metric': 'acc', 'task type': 'multiclass classification'}
dataset_dict[name]['download_name'] = 'BALBc_no2_pb_minRadiusAvg'
dataset_dict[name]['version'] = 1
# usually use the sync and share link; for test, debug and development you can also use a local file link
dataset_dict[name]['url'] = 'https://syncandshare.lrz.de/dl/fiJPVE5nL72VuwKoqPVNVXSq/BALBc_no2_pb_minRadiusAvg.zip'
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]['add_inverse_edge'] = False
dataset_dict[name]['has_node_attr'] = True
dataset_dict[name]['has_edge_attr'] = True
dataset_dict[name]['split'] = 'random'
dataset_dict[name]['additional node files'] = 'None'
dataset_dict[name]['additional edge files'] = 'None'
dataset_dict[name]['is hetero'] = False
dataset_dict[name]['binary'] = True

### add meta-information about product category prediction task
name = 'ogbn-BALBc_no3_pb_minRadiusAvg'
dataset_dict[name] = {'num tasks': 1, 'num classes': 3, 'eval metric': 'acc', 'task type': 'multiclass classification'}
dataset_dict[name]['download_name'] = 'BALBc_no3_pb_minRadiusAvg'
dataset_dict[name]['version'] = 1
# usually use the sync and share link; for test, debug and development you can also use a local file link
dataset_dict[name]['url'] = 'https://syncandshare.lrz.de/dl/fi7tx5cM1y5mw4ibsdEQZHSb/BALBc_no3_pb_minRadiusAvg.zip'
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]['add_inverse_edge'] = False
dataset_dict[name]['has_node_attr'] = True
dataset_dict[name]['has_edge_attr'] = True
dataset_dict[name]['split'] = 'random'
dataset_dict[name]['additional node files'] = 'None'
dataset_dict[name]['additional edge files'] = 'None'
dataset_dict[name]['is hetero'] = False
dataset_dict[name]['binary'] = True

### add meta-information about product category prediction task
name = 'ogbn-C57BL_6_no1_pb_minRadiusAvg'
dataset_dict[name] = {'num tasks': 1, 'num classes': 3, 'eval metric': 'acc', 'task type': 'multiclass classification'}
dataset_dict[name]['download_name'] = 'C57BL_6_no1_pb_minRadiusAvg'
dataset_dict[name]['version'] = 1
# usually use the sync and share link; for test, debug and development you can also use a local file link
dataset_dict[name]['url'] = 'https://syncandshare.lrz.de/dl/fiFG3knmCpp3imbrgN1PbfVR/C57BL_6_no1_pb_minRadiusAvg.zip'
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]['add_inverse_edge'] = False
dataset_dict[name]['has_node_attr'] = True
dataset_dict[name]['has_edge_attr'] = True
dataset_dict[name]['split'] = 'random'
dataset_dict[name]['additional node files'] = 'None'
dataset_dict[name]['additional edge files'] = 'None'
dataset_dict[name]['is hetero'] = False
dataset_dict[name]['binary'] = True

### add meta-information about product category prediction task
name = 'ogbn-C57BL_6_no2_pb_minRadiusAvg'
dataset_dict[name] = {'num tasks': 1, 'num classes': 3, 'eval metric': 'acc', 'task type': 'multiclass classification'}
dataset_dict[name]['download_name'] = 'C57BL_6_no2_pb_minRadiusAvg'
dataset_dict[name]['version'] = 1
# usually use the sync and share link; for test, debug and development you can also use a local file link
dataset_dict[name]['url'] = 'https://syncandshare.lrz.de/dl/fiUgDnReXHP4XRkRboFMJ34z/C57BL_6_no2_pb_minRadiusAvg.zip'
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]['add_inverse_edge'] = False
dataset_dict[name]['has_node_attr'] = True
dataset_dict[name]['has_edge_attr'] = True
dataset_dict[name]['split'] = 'random'
dataset_dict[name]['additional node files'] = 'None'
dataset_dict[name]['additional edge files'] = 'None'
dataset_dict[name]['is hetero'] = False
dataset_dict[name]['binary'] = True

### add meta-information about product category prediction task
name = 'ogbn-C57BL_6_no3_pb_minRadiusAvg'
dataset_dict[name] = {'num tasks': 1, 'num classes': 3, 'eval metric': 'acc', 'task type': 'multiclass classification'}
dataset_dict[name]['download_name'] = 'C57BL_6_no3_pb_minRadiusAvg'
dataset_dict[name]['version'] = 1
# usually use the sync and share link; for test, debug and development you can also use a local file link
dataset_dict[name]['url'] = 'https://syncandshare.lrz.de/dl/fi31Qu1YR8C4aTQinBYAcZGw/C57BL_6_no3_pb_minRadiusAvg.zip'
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]['add_inverse_edge'] = False
dataset_dict[name]['has_node_attr'] = True
dataset_dict[name]['has_edge_attr'] = True
dataset_dict[name]['split'] = 'random'
dataset_dict[name]['additional node files'] = 'None'
dataset_dict[name]['additional edge files'] = 'None'
dataset_dict[name]['is hetero'] = False
dataset_dict[name]['binary'] = True

### add meta-information about product category prediction task
name = 'ogbn-CD1_E_no1_pb_minRadiusAvg'
dataset_dict[name] = {'num tasks': 1, 'num classes': 3, 'eval metric': 'acc', 'task type': 'multiclass classification'}
dataset_dict[name]['download_name'] = 'CD1_E_no1_pb_minRadiusAvg'
dataset_dict[name]['version'] = 1
# usually use the sync and share link; for test, debug and development you can also use a local file link
dataset_dict[name]['url'] = 'https://syncandshare.lrz.de/dl/fiTqGqPwF4YPYSHXi45VY6ev/CD1_E_no1_pb_minRadiusAvg.zip'
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]['add_inverse_edge'] = False
dataset_dict[name]['has_node_attr'] = True
dataset_dict[name]['has_edge_attr'] = True
dataset_dict[name]['split'] = 'random'
dataset_dict[name]['additional node files'] = 'None'
dataset_dict[name]['additional edge files'] = 'None'
dataset_dict[name]['is hetero'] = False
dataset_dict[name]['binary'] = True

### add meta-information about product category prediction task
name = 'ogbn-CD1_E_no2_pb_minRadiusAvg'
dataset_dict[name] = {'num tasks': 1, 'num classes': 3, 'eval metric': 'acc', 'task type': 'multiclass classification'}
dataset_dict[name]['download_name'] = 'CD1_E_no2_pb_minRadiusAvg'
dataset_dict[name]['version'] = 1
# usually use the sync and share link; for test, debug and development you can also use a local file link
dataset_dict[name]['url'] = 'https://syncandshare.lrz.de/dl/fi7pYjyE7YxcvRBoVvMymvS2/CD1_E_no2_pb_minRadiusAvg.zip'
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]['add_inverse_edge'] = False
dataset_dict[name]['has_node_attr'] = True
dataset_dict[name]['has_edge_attr'] = True
dataset_dict[name]['split'] = 'random'
dataset_dict[name]['additional node files'] = 'None'
dataset_dict[name]['additional edge files'] = 'None'
dataset_dict[name]['is hetero'] = False
dataset_dict[name]['binary'] = True

### add meta-information about product category prediction task
name = 'ogbn-CD1_E_no3_pb_minRadiusAvg'
dataset_dict[name] = {'num tasks': 1, 'num classes': 3, 'eval metric': 'acc', 'task type': 'multiclass classification'}
dataset_dict[name]['download_name'] = 'CD1_E_no3_pb_minRadiusAvg'
dataset_dict[name]['version'] = 1
# usually use the sync and share link; for test, debug and development you can also use a local file link
dataset_dict[name]['url'] = 'https://syncandshare.lrz.de/dl/fiDkZv2o6N5qaw37Gd6tbjkg/CD1_E_no3_pb_minRadiusAvg.zip'
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]['add_inverse_edge'] = False
dataset_dict[name]['has_node_attr'] = True
dataset_dict[name]['has_edge_attr'] = True
dataset_dict[name]['split'] = 'random'
dataset_dict[name]['additional node files'] = 'None'
dataset_dict[name]['additional edge files'] = 'None'
dataset_dict[name]['is hetero'] = False
dataset_dict[name]['binary'] = True


### add meta-information about product category prediction task
name = 'ogbn-node_vessap_roi3_pb_minRadiusAvg'
dataset_dict[name] = {'num tasks': 1, 'num classes': 3, 'eval metric': 'acc', 'task type': 'multiclass classification'}
dataset_dict[name]['download_name'] = 'node_vessap_roi3_pb_minRadiusAvg'
dataset_dict[name]['version'] = 1
# usually use the sync and share link; for test, debug and development you can also use a local file link
dataset_dict[name]['url'] = 'https://syncandshare.lrz.de/dl/fiC9VCUydPr2uknrHpShXhZP/node_vessap_roi3_pb_minRadiusAvg.zip'
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]['add_inverse_edge'] = False
dataset_dict[name]['has_node_attr'] = True
dataset_dict[name]['has_edge_attr'] = True
dataset_dict[name]['split'] = 'random'
dataset_dict[name]['additional node files'] = 'None'
dataset_dict[name]['additional edge files'] = 'None'
dataset_dict[name]['is hetero'] = False
dataset_dict[name]['binary'] = True


df = pd.DataFrame(dataset_dict)
# saving the dataframe 
df.to_csv('master.csv')