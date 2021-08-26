import pandas as pd
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Preprocess edge list')
parser.add_argument('--atlas_groups', help='Atlas Groups - List to be processed.', required=True)
parser.add_argument('--atlas_nodes', help='Atlas Node List to be processed.', required=True)
args = parser.parse_args()

df_atlas_groups = pd.read_csv(args.atlas_groups,sep=',',usecols=['GroupAcronym','GroupedAcronyms'])
df_atlas = pd.read_csv(args.atlas_nodes,sep=',',usecols=['Region_Acronym'])

mapping = {}
mapping['bgr'] = 'bgr'
mapping['nan'] = 'bgr'
mapping['root'] = 'root'
mapping['fiber tracts'] = 'fiber tracts'
mapping['CUL4, 5'] = 'CUL4'

for j in tqdm(range(0, len(df_atlas_groups.index))):
    # print(j)
    # get all items
    items = [s.replace(",", "") for s in df_atlas_groups.iloc[j]['GroupedAcronyms'].split()]
    acronym = df_atlas_groups.iloc[j]['GroupAcronym']
    for item in items:
        mapping[str(item)] = acronym

for k in tqdm(range(0, len(df_atlas.index))):
    #print(str(df_atlas.at[k, 'Region_Acronym']))
    df_atlas.at[k, 'Region_Acronym'] = mapping[str(df_atlas.at[k, 'Region_Acronym'])]


df_one_hot = pd.get_dummies(df_atlas, prefix =['Region_Acronym'],columns = ['Region_Acronym']) # assign column names

path = args.atlas_nodes.split('.csv')[0] + "_atlas_processed.csv"
df_one_hot.to_csv(path,sep=";",index=False)
