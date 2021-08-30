### script for writing meta information of datasets into master.csv
### for link property prediction datasets.
import pandas as pd

dataset_dict = {}
dataset_list = []

### add meta-information about protein function prediction task
name = 'ogbl-ppa'
dataset_dict[name] = {'eval metric': 'hits@100', 'task type': 'link prediction'}
dataset_dict[name]['download_name'] = 'ppassoc'
dataset_dict[name]['version'] = 1
dataset_dict[name]['url'] = 'http://snap.stanford.edu/ogb/data/linkproppred/'+dataset_dict[name]['download_name']+'.zip'
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]['add_inverse_edge'] = True 
dataset_dict[name]['has_node_attr'] = True
dataset_dict[name]['has_edge_attr'] = False
dataset_dict[name]['split'] = 'throughput'
dataset_dict[name]['additional node files'] = 'None'
dataset_dict[name]['additional edge files'] = 'None'
dataset_dict[name]['is hetero'] = False
dataset_dict[name]['binary'] = False

### add meta-information about author collaboration prediction task
name = 'ogbl-collab'
dataset_dict[name] = {'eval metric': 'hits@50', 'task type': 'link prediction'}
dataset_dict[name]['download_name'] = 'collab'
dataset_dict[name]['version'] = 1
dataset_dict[name]['url'] = 'http://snap.stanford.edu/ogb/data/linkproppred/'+dataset_dict[name]['download_name']+'.zip'
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]['add_inverse_edge'] = True 
dataset_dict[name]['has_node_attr'] = True
dataset_dict[name]['has_edge_attr'] = False
dataset_dict[name]['split'] = 'time'
dataset_dict[name]['additional node files'] = 'None'
dataset_dict[name]['additional edge files'] = 'edge_weight,edge_year'
dataset_dict[name]['is hetero'] = False
dataset_dict[name]['binary'] = False

### add meta-information about paper citation recommendation task
### ogbl-citation is depreciated due to the negative sample bug (Dec 25, 2020)
name = 'ogbl-citation2'
dataset_dict[name] = {'eval metric': 'mrr', 'task type': 'link prediction'}
dataset_dict[name]['download_name'] = 'citation-v2'
dataset_dict[name]['version'] = 1
dataset_dict[name]['url'] = 'http://snap.stanford.edu/ogb/data/linkproppred/'+dataset_dict[name]['download_name']+'.zip'
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]['add_inverse_edge'] = False 
dataset_dict[name]['has_node_attr'] = True
dataset_dict[name]['has_edge_attr'] = False
dataset_dict[name]['split'] = 'time'
dataset_dict[name]['additional node files'] = 'node_year'
dataset_dict[name]['additional edge files'] = 'None'
dataset_dict[name]['is hetero'] = False
dataset_dict[name]['binary'] = False

### add meta-information about wikidata knowledge graph completion task
### ogbl-wikikg is depreciated due to the negative sample bug (Dec 25, 2020)
name = 'ogbl-wikikg2'
dataset_dict[name] = {'eval metric': 'mrr', 'task type': 'KG completion'}
dataset_dict[name]['download_name'] = 'wikikg-v2'
dataset_dict[name]['version'] = 1
dataset_dict[name]['url'] = 'http://snap.stanford.edu/ogb/data/linkproppred/'+dataset_dict[name]['download_name']+'.zip'
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]['add_inverse_edge'] = False 
dataset_dict[name]['has_node_attr'] = False
dataset_dict[name]['has_edge_attr'] = False
dataset_dict[name]['split'] = 'time'
dataset_dict[name]['additional node files'] = 'None'
dataset_dict[name]['additional edge files'] = 'edge_reltype'
dataset_dict[name]['is hetero'] = False
dataset_dict[name]['binary'] = False

### add meta-information about drug drug interaction prediction task
name = 'ogbl-ddi'
dataset_dict[name] = {'eval metric': 'hits@20', 'task type': 'link prediction'}
dataset_dict[name]['download_name'] = 'ddi'
dataset_dict[name]['version'] = 1
dataset_dict[name]['url'] = 'http://snap.stanford.edu/ogb/data/linkproppred/'+dataset_dict[name]['download_name']+'.zip'
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]['add_inverse_edge'] = True 
dataset_dict[name]['has_node_attr'] = False
dataset_dict[name]['has_edge_attr'] = False
dataset_dict[name]['split'] = 'target'
dataset_dict[name]['additional node files'] = 'None'
dataset_dict[name]['additional edge files'] = 'None'
dataset_dict[name]['is hetero'] = False
dataset_dict[name]['binary'] = False

### add meta-information about biological knowledge graph completion task
name = 'ogbl-biokg'
dataset_dict[name] = {'eval metric': 'mrr', 'task type': 'KG completion'}
dataset_dict[name]['download_name'] = 'biokg'
dataset_dict[name]['version'] = 1
dataset_dict[name]['url'] = 'http://snap.stanford.edu/ogb/data/linkproppred/'+dataset_dict[name]['download_name']+'.zip'
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]['add_inverse_edge'] = True 
dataset_dict[name]['has_node_attr'] = False
dataset_dict[name]['has_edge_attr'] = False
dataset_dict[name]['split'] = 'random'
dataset_dict[name]['additional node files'] = 'None'
dataset_dict[name]['additional edge files'] = 'edge_reltype'
dataset_dict[name]['is hetero'] = True
dataset_dict[name]['binary'] = False


#################################################################################################################################
#                                                                                                                               #
#                                                   TUM Vessap ROIs for Hyperparameter Tuning                                                     #
#                                                                                                                               #
#################################################################################################################################


### add meta-information about vessap roi 3
name = 'ogbl-link_vessap_roi3_spatial_no_edge_attr'
dataset_dict[name] = {'eval metric': 'mrr', 'task type': 'link prediction'}
dataset_dict[name]['download_name'] = 'link_vessap_roi3_spatial_no_edge_attr'
dataset_dict[name]['version'] = 1
dataset_dict[name]['url'] = 'https://syncandshare.lrz.de/dl/fiX7PfeewXsKMCZTUTVFE6jD/link_vessap_roi3_spatial_no_edge_attr.zip'
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]['add_inverse_edge'] = False # c.f. https://github.com/snap-stanford/ogb/issues/241
dataset_dict[name]['has_node_attr'] = True
dataset_dict[name]['has_edge_attr'] = False
dataset_dict[name]['split'] = 'spatial'
dataset_dict[name]['additional node files'] = 'None'
dataset_dict[name]['additional edge files'] = 'None'
dataset_dict[name]['is hetero'] = False
dataset_dict[name]['binary'] = True


#################################################################################################################################
#                                                                                                                               #
#                                                   TUM WHOLE BRAIN VESSAP GRAPHS                                               #
#                                                                                                                               #
#################################################################################################################################

### add meta-information about BALBc_no1
name = 'ogbl-BALBc_no1_spatial_edge_attr'
dataset_dict[name] = {'eval metric': 'mrr', 'task type': 'link prediction'}
dataset_dict[name]['download_name'] = 'BALBc_no1_spatial_edge_attr'
dataset_dict[name]['version'] = 1
dataset_dict[name]['url'] = 'https://syncandshare.lrz.de/dl/fiHHnVwcmQsnZXfkKFLeo4Le/BALBc_no1_spatial_no_edge_attr.zip'
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]['add_inverse_edge'] = False # c.f. https://github.com/snap-stanford/ogb/issues/241
dataset_dict[name]['has_node_attr'] = True
dataset_dict[name]['has_edge_attr'] = True
dataset_dict[name]['split'] = 'spatial'
dataset_dict[name]['additional node files'] = 'None'
dataset_dict[name]['additional edge files'] = 'None'
dataset_dict[name]['is hetero'] = False
dataset_dict[name]['binary'] = True

### add meta-information about BALBc_no1
name = 'ogbl-BALBc_no1_spatial_no_edge_attr'
dataset_dict[name] = {'eval metric': 'mrr', 'task type': 'link prediction'}
dataset_dict[name]['download_name'] = 'BALBc_no1_spatial_no_edge_attr'
dataset_dict[name]['version'] = 1
dataset_dict[name]['url'] = ''
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]['add_inverse_edge'] = False # c.f. https://github.com/snap-stanford/ogb/issues/241
dataset_dict[name]['has_node_attr'] = True
dataset_dict[name]['has_edge_attr'] = False
dataset_dict[name]['split'] = 'spatial'
dataset_dict[name]['additional node files'] = 'None'
dataset_dict[name]['additional edge files'] = 'None'
dataset_dict[name]['is hetero'] = False
dataset_dict[name]['binary'] = True

### add meta-information about BALBc_no2
name = 'ogbl-link_vessap_BALBc_no2_spatial_edge_attr'
dataset_dict[name] = {'eval metric': 'mrr', 'task type': 'link prediction'}
dataset_dict[name]['download_name'] = 'link_vessap_BALBc_no2_spatial_edge_attr'
dataset_dict[name]['version'] = 1
dataset_dict[name]['url'] = ''
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]['add_inverse_edge'] = False # c.f. https://github.com/snap-stanford/ogb/issues/241
dataset_dict[name]['has_node_attr'] = True
dataset_dict[name]['has_edge_attr'] = True
dataset_dict[name]['split'] = 'spatial'
dataset_dict[name]['additional node files'] = 'None'
dataset_dict[name]['additional edge files'] = 'None'
dataset_dict[name]['is hetero'] = False
dataset_dict[name]['binary'] = True

### add meta-information about BALBc_no2
name = 'ogbl-link_vessap_BALBc_no2_spatial_no_edge_attr'
dataset_dict[name] = {'eval metric': 'mrr', 'task type': 'link prediction'}
dataset_dict[name]['download_name'] = 'link_vessap_BALBc_no2_spatial_no_edge_attr'
dataset_dict[name]['version'] = 1
dataset_dict[name]['url'] = ''
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]['add_inverse_edge'] = False # c.f. https://github.com/snap-stanford/ogb/issues/241
dataset_dict[name]['has_node_attr'] = True
dataset_dict[name]['has_edge_attr'] = False
dataset_dict[name]['split'] = 'spatial'
dataset_dict[name]['additional node files'] = 'None'
dataset_dict[name]['additional edge files'] = 'None'
dataset_dict[name]['is hetero'] = False
dataset_dict[name]['binary'] = True

### add meta-information about BALBc_no3
name = 'ogbl-link_vessap_BALBc_no3_spatial_edge_attr'
dataset_dict[name] = {'eval metric': 'mrr', 'task type': 'link prediction'}
dataset_dict[name]['download_name'] = 'link_vessap_BALBc_no3_spatial_edge_attr'
dataset_dict[name]['version'] = 1
dataset_dict[name]['url'] = ''
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]['add_inverse_edge'] = False # c.f. https://github.com/snap-stanford/ogb/issues/241
dataset_dict[name]['has_node_attr'] = True
dataset_dict[name]['has_edge_attr'] = True
dataset_dict[name]['split'] = 'spatial'
dataset_dict[name]['additional node files'] = 'None'
dataset_dict[name]['additional edge files'] = 'None'
dataset_dict[name]['is hetero'] = False
dataset_dict[name]['binary'] = True

### add meta-information about BALBc_no3
name = 'ogbl-link_vessap_BALBc_no3_spatial_no_edge_attr'
dataset_dict[name] = {'eval metric': 'mrr', 'task type': 'link prediction'}
dataset_dict[name]['download_name'] = 'link_vessap_BALBc_no3_spatial_no_edge_attr'
dataset_dict[name]['version'] = 1
dataset_dict[name]['url'] = ''
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]['add_inverse_edge'] = False # c.f. https://github.com/snap-stanford/ogb/issues/241
dataset_dict[name]['has_node_attr'] = True
dataset_dict[name]['has_edge_attr'] = False
dataset_dict[name]['split'] = 'spatial'
dataset_dict[name]['additional node files'] = 'None'
dataset_dict[name]['additional edge files'] = 'None'
dataset_dict[name]['is hetero'] = False
dataset_dict[name]['binary'] = True

########################################################################################################################################

### add meta-information about BL6J_no1
name = 'ogbl-link_vessap_BL6J_no1_spatial_edge_attr'
dataset_dict[name] = {'eval metric': 'mrr', 'task type': 'link prediction'}
dataset_dict[name]['download_name'] = 'link_vessap_BL6J_no1_spatial_edge_attr'
dataset_dict[name]['version'] = 1
dataset_dict[name]['url'] = ''
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]['add_inverse_edge'] = False # c.f. https://github.com/snap-stanford/ogb/issues/241
dataset_dict[name]['has_node_attr'] = True
dataset_dict[name]['has_edge_attr'] = True
dataset_dict[name]['split'] = 'spatial'
dataset_dict[name]['additional node files'] = 'None'
dataset_dict[name]['additional edge files'] = 'None'
dataset_dict[name]['is hetero'] = False
dataset_dict[name]['binary'] = True

### add meta-information about BL6J_no1
name = 'ogbl-link_vessap_BL6J_no1_spatial_no_edge_attr'
dataset_dict[name] = {'eval metric': 'mrr', 'task type': 'link prediction'}
dataset_dict[name]['download_name'] = 'link_vessap_BL6J_no1_spatial_no_edge_attr'
dataset_dict[name]['version'] = 1
dataset_dict[name]['url'] = ''
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]['add_inverse_edge'] = False # c.f. https://github.com/snap-stanford/ogb/issues/241
dataset_dict[name]['has_node_attr'] = True
dataset_dict[name]['has_edge_attr'] = False
dataset_dict[name]['split'] = 'spatial'
dataset_dict[name]['additional node files'] = 'None'
dataset_dict[name]['additional edge files'] = 'None'
dataset_dict[name]['is hetero'] = False
dataset_dict[name]['binary'] = True

### add meta-information about BL6J_no2
name = 'ogbl-link_vessap_BL6J_no2_spatial_edge_attr'
dataset_dict[name] = {'eval metric': 'mrr', 'task type': 'link prediction'}
dataset_dict[name]['download_name'] = 'link_vessap_BL6J_no2_spatial_edge_attr'
dataset_dict[name]['version'] = 1
dataset_dict[name]['url'] = ''
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]['add_inverse_edge'] = False # c.f. https://github.com/snap-stanford/ogb/issues/241
dataset_dict[name]['has_node_attr'] = True
dataset_dict[name]['has_edge_attr'] = True
dataset_dict[name]['split'] = 'spatial'
dataset_dict[name]['additional node files'] = 'None'
dataset_dict[name]['additional edge files'] = 'None'
dataset_dict[name]['is hetero'] = False
dataset_dict[name]['binary'] = True

### add meta-information about BL6J_no2
name = 'ogbl-link_vessap_BL6J_no2_spatial_no_edge_attr'
dataset_dict[name] = {'eval metric': 'mrr', 'task type': 'link prediction'}
dataset_dict[name]['download_name'] = 'link_vessap_BL6J_no2_spatial_no_edge_attr'
dataset_dict[name]['version'] = 1
dataset_dict[name]['url'] = ''
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]['add_inverse_edge'] = False # c.f. https://github.com/snap-stanford/ogb/issues/241
dataset_dict[name]['has_node_attr'] = True
dataset_dict[name]['has_edge_attr'] = False
dataset_dict[name]['split'] = 'spatial'
dataset_dict[name]['additional node files'] = 'None'
dataset_dict[name]['additional edge files'] = 'None'
dataset_dict[name]['is hetero'] = False
dataset_dict[name]['binary'] = True

### add meta-information about BL6J_no3
name = 'ogbl-link_vessap_BL6J_no3_spatial_edge_attr'
dataset_dict[name] = {'eval metric': 'mrr', 'task type': 'link prediction'}
dataset_dict[name]['download_name'] = 'link_vessap_BL6J_no3_spatial_edge_attr'
dataset_dict[name]['version'] = 1
dataset_dict[name]['url'] = ''
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]['add_inverse_edge'] = False # c.f. https://github.com/snap-stanford/ogb/issues/241
dataset_dict[name]['has_node_attr'] = True
dataset_dict[name]['has_edge_attr'] = True
dataset_dict[name]['split'] = 'spatial'
dataset_dict[name]['additional node files'] = 'None'
dataset_dict[name]['additional edge files'] = 'None'
dataset_dict[name]['is hetero'] = False
dataset_dict[name]['binary'] = True

### add meta-information about BL6J_no3
name = 'ogbl-link_vessap_BL6J_no3_spatial_no_edge_attr'
dataset_dict[name] = {'eval metric': 'mrr', 'task type': 'link prediction'}
dataset_dict[name]['download_name'] = 'link_vessap_BL6J_no3_spatial_no_edge_attr'
dataset_dict[name]['version'] = 1
dataset_dict[name]['url'] = ''
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]['add_inverse_edge'] = False # c.f. https://github.com/snap-stanford/ogb/issues/241
dataset_dict[name]['has_node_attr'] = True
dataset_dict[name]['has_edge_attr'] = False
dataset_dict[name]['split'] = 'spatial'
dataset_dict[name]['additional node files'] = 'None'
dataset_dict[name]['additional edge files'] = 'None'
dataset_dict[name]['is hetero'] = False
dataset_dict[name]['binary'] = True

############################################################################################################################################

### add meta-information about CD1-E_no1
name = 'ogbl-link_vessap_CD1-E_no1_spatial_edge_attr'
dataset_dict[name] = {'eval metric': 'mrr', 'task type': 'link prediction'}
dataset_dict[name]['download_name'] = 'link_vessap_CD1-E_no1_spatial_edge_attr'
dataset_dict[name]['version'] = 1
dataset_dict[name]['url'] = ''
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]['add_inverse_edge'] = False # c.f. https://github.com/snap-stanford/ogb/issues/241
dataset_dict[name]['has_node_attr'] = True
dataset_dict[name]['has_edge_attr'] = True
dataset_dict[name]['split'] = 'spatial'
dataset_dict[name]['additional node files'] = 'None'
dataset_dict[name]['additional edge files'] = 'None'
dataset_dict[name]['is hetero'] = False
dataset_dict[name]['binary'] = True

### add meta-information about CD1-E_no1
name = 'ogbl-link_vessap_CD1-E_no1_spatial_no_edge_attr'
dataset_dict[name] = {'eval metric': 'mrr', 'task type': 'link prediction'}
dataset_dict[name]['download_name'] = 'link_vessap_CD1-E_no1_spatial_no_edge_attr'
dataset_dict[name]['version'] = 1
dataset_dict[name]['url'] = ''
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]['add_inverse_edge'] = False # c.f. https://github.com/snap-stanford/ogb/issues/241
dataset_dict[name]['has_node_attr'] = True
dataset_dict[name]['has_edge_attr'] = False
dataset_dict[name]['split'] = 'spatial'
dataset_dict[name]['additional node files'] = 'None'
dataset_dict[name]['additional edge files'] = 'None'
dataset_dict[name]['is hetero'] = False
dataset_dict[name]['binary'] = True

### add meta-information about CD1-E_no2
name = 'ogbl-link_vessap_CD1-E_no2_spatial_edge_attr'
dataset_dict[name] = {'eval metric': 'mrr', 'task type': 'link prediction'}
dataset_dict[name]['download_name'] = 'link_vessap_CD1-E_no2_spatial_edge_attr'
dataset_dict[name]['version'] = 1
dataset_dict[name]['url'] = ''
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]['add_inverse_edge'] = False # c.f. https://github.com/snap-stanford/ogb/issues/241
dataset_dict[name]['has_node_attr'] = True
dataset_dict[name]['has_edge_attr'] = True
dataset_dict[name]['split'] = 'spatial'
dataset_dict[name]['additional node files'] = 'None'
dataset_dict[name]['additional edge files'] = 'None'
dataset_dict[name]['is hetero'] = False
dataset_dict[name]['binary'] = True

### add meta-information about CD1-E_no2
name = 'ogbl-link_vessap_CD1-E_no2_spatial_no_edge_attr'
dataset_dict[name] = {'eval metric': 'mrr', 'task type': 'link prediction'}
dataset_dict[name]['download_name'] = 'link_vessap_CD1-E_no2_spatial_no_edge_attr'
dataset_dict[name]['version'] = 1
dataset_dict[name]['url'] = ''
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]['add_inverse_edge'] = False # c.f. https://github.com/snap-stanford/ogb/issues/241
dataset_dict[name]['has_node_attr'] = True
dataset_dict[name]['has_edge_attr'] = False
dataset_dict[name]['split'] = 'spatial'
dataset_dict[name]['additional node files'] = 'None'
dataset_dict[name]['additional edge files'] = 'None'
dataset_dict[name]['is hetero'] = False
dataset_dict[name]['binary'] = True

### add meta-information about CD1-E_no3
name = 'ogbl-link_vessap_CD1-E_no3_spatial_edge_attr'
dataset_dict[name] = {'eval metric': 'mrr', 'task type': 'link prediction'}
dataset_dict[name]['download_name'] = 'link_vessap_CD1-E_no3_spatial_edge_attr'
dataset_dict[name]['version'] = 1
dataset_dict[name]['url'] = ''
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]['add_inverse_edge'] = False # c.f. https://github.com/snap-stanford/ogb/issues/241
dataset_dict[name]['has_node_attr'] = True
dataset_dict[name]['has_edge_attr'] = True
dataset_dict[name]['split'] = 'spatial'
dataset_dict[name]['additional node files'] = 'None'
dataset_dict[name]['additional edge files'] = 'None'
dataset_dict[name]['is hetero'] = False
dataset_dict[name]['binary'] = True

### add meta-information about CD1-E_no3
name = 'ogbl-link_vessap_CD1-E_no3_spatial_no_edge_attr'
dataset_dict[name] = {'eval metric': 'mrr', 'task type': 'link prediction'}
dataset_dict[name]['download_name'] = 'link_vessap_CD1-E_no3_spatial_no_edge_attr'
dataset_dict[name]['version'] = 1
dataset_dict[name]['url'] = ''
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]['add_inverse_edge'] = False # c.f. https://github.com/snap-stanford/ogb/issues/241
dataset_dict[name]['has_node_attr'] = True
dataset_dict[name]['has_edge_attr'] = False
dataset_dict[name]['split'] = 'spatial'
dataset_dict[name]['additional node files'] = 'None'
dataset_dict[name]['additional edge files'] = 'None'
dataset_dict[name]['is hetero'] = False
dataset_dict[name]['binary'] = True

df = pd.DataFrame(dataset_dict)
# saving the dataframe 
df.to_csv('master.csv')


