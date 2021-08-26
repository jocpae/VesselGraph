# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 15:01:03 2021

@author: mtodorov, adapted by Julian McGinnis
"""
import numpy as np
import csv
import nibabel as nib
from tqdm import tqdm
import pandas as pd
import datetime
import argparse
from xml.etree import ElementTree as ET
import io

parser = argparse.ArgumentParser(description='annotates voreen node file with Allen Mouse Brain Atlas.')
parser.add_argument('-n','--node_list', help='Name of the node list csv file.', required=True)
parser.add_argument('-a','--atlas_result', help='Name of the atlas registration results as .nii.gz', required=True)
parser.add_argument('-o','--onthology_file', help='Ontology XML File of the Alan Brain Atlas', required=True)

# read the arguments
args = vars(parser.parse_args())

OntologyFilePath = "AllenMouseCCFv3_ontology_22Feb2021.xml"
LabelFilePath = "BALBC-no1_iso3um_stitched_atlas_registration_result.nii.gz"
CSVFilePath      = "BALBc-no1_iso3um_graph_nodes.csv"

OntologyFilePath = args['onthology_file']
LabelFilePath  = args['atlas_result']
CSVFilePath   = args['node_list']

columnNames=['Label', 'VoxelSum', 'RegionSum']
excludeRegionByColorHexList=['000000', 'FFFFFF', 'BFDAE3', 'B0F0FF', 'B0FFB8', '70FF70', '70FF71', '82C7AE', '4BB547', '58BA48', '56B84B', '33B932', 'ECE754', '7F2E7E']
VoxelCorrectionFactor = 1.625 * 1.625 * 3


def parseOntologyXML(ontologyInput=None):
    
    if ontologyInput == None: raise Exception("An Allen CCF ontology XML file must be provided.")
    print(datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + " Parsing ontology XML" )  
        
    outputFileColumns = ['id','name', 'acronym', 'red', 'green', 'blue', 'structure_order', 'parent_id', 'parent_acronym', 'color-hex-triplet']
    
    with io.open(ontologyInput, 'r', encoding='utf-8-sig') as f:
        contents = f.read()
        root = ET.fromstring(contents)
    
    ontologyParsed = []
    ontologyParsed.append(tuple(outputFileColumns))

    row=[0,'background','bgr',0,0,0,0,'None','None','000000']
    ontologyParsed.append(tuple(row))
    
    for atlasStructure in root.iter('structure'):
        
        #identify root structure and print its acronym
        structures = root.iter('structure')
        for tmp in structures:
            RandAttribute = tmp.findall('id')[0].text
            if RandAttribute == atlasStructure.find('parent-structure-id').text:
                ci_name = tmp.findall('acronym')[0].text               
                
                
        if atlasStructure.find('id-original') == None:
            structureId = atlasStructure.find('id').text
        else:
            structureId = atlasStructure.find('id-original').text
        #structureId = atlasStructure.find('id').text
            
        if int(structureId) == 997:
            ci_name='"root"'
            print(datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + " parent of ID 997 is always mapped to root" )  
            
        if int(structureId) == 312782566:
            structureId=312782560
            print(datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + " mapping ID 312782566 --> 312782560 (only the latter exists is in the annotation NRRD)" )  
            
        if int(structureId) == 614454277:
            structureId=614454272
            print(datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + " mapping ID 614454277 --> 614454272 (only the latter exists is in the annotation NRRD)" )
                
        row=[int(structureId) ,
              atlasStructure.find('name').text , 
              atlasStructure.find('acronym').text.replace('"','') , 
              tuple(int(atlasStructure.find('color-hex-triplet').text[i:i+2], 16) for i in (0, 2, 4))[0], 
              tuple(int(atlasStructure.find('color-hex-triplet').text[i:i+2], 16) for i in (0, 2, 4))[1],
              tuple(int(atlasStructure.find('color-hex-triplet').text[i:i+2], 16) for i in (0, 2, 4))[2],
              atlasStructure.find('graph-order').text,
              atlasStructure.find('parent-structure-id').text,             
              ci_name,             
              atlasStructure.find('color-hex-triplet').text ]
        ontologyParsed.append(tuple(row))
        
    ontologyDF=pd.DataFrame.from_records(ontologyParsed[1:], columns=outputFileColumns)
    print(datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ' Parsing finished, found ' + str(ontologyDF['id'].unique().shape[0]) + ' unique IDs' )
    
    return ontologyDF


def collapseToColorGroup(ElementsList, excludeRegions=None):
    if ElementsList==None: raise Exception("ElementsList not specified!")
    
    tmp = pd.DataFrame(ElementsList) 
    
    
    groupTemplate=pd.DataFrame(columns=["ColorGroup", "GroupName", "GroupAcronym", "GroupedAcronyms", "BlobCount", "BlobVoxelSum", "BlobVolumeSum (um3)", "BlobVoxelAve", "BlobVolumeAve (um3)" ])
    groupTemplate['ColorGroup'] = ontologyDF['color-hex-triplet'].unique().tolist()
    for i, row in groupTemplate.iterrows():
        acronymList = ontologyDF[ontologyDF['color-hex-triplet'] == groupTemplate.at[i,'ColorGroup']]['acronym'].tolist()
        groupTemplate.at[i,'GroupedAcronyms']        = ', '.join(acronymList)
        groupTemplate.at[i,'GroupName']              = ontologyDF[ontologyDF['acronym']==acronymList[0]]['name'].to_string(index=False).strip()
        groupTemplate.at[i,'GroupAcronym']           = acronymList[0]
        
        groupTemplate.at[i,'BlobCount']              = tmp[tmp["Region_ColorHex"]==groupTemplate.at[i,'ColorGroup']]["Blob_NumOfVoxels"].count()
        groupTemplate.at[i,'BlobVoxelSum']           = tmp[tmp["Region_ColorHex"]==groupTemplate.at[i,'ColorGroup']]["Blob_NumOfVoxels"].to_numpy().astype(np.int).sum()
        groupTemplate.at[i,'BlobVolumeSum (um3)']    = tmp[tmp["Region_ColorHex"]==groupTemplate.at[i,'ColorGroup']]["Blob_NumOfVoxels"].to_numpy().astype(np.int).sum()  * VoxelCorrectionFactor
        groupTemplate.at[i,'BlobVoxelAve']           = tmp[tmp["Region_ColorHex"]==groupTemplate.at[i,'ColorGroup']]["Blob_NumOfVoxels"].to_numpy().astype(np.int).mean()
        groupTemplate.at[i,'BlobVolumeAve (um3)']    = tmp[tmp["Region_ColorHex"]==groupTemplate.at[i,'ColorGroup']]["Blob_NumOfVoxels"].to_numpy().astype(np.int).mean() * VoxelCorrectionFactor
        
    if not excludeRegions == None:
        groupTemplate  = groupTemplate[~groupTemplate['ColorGroup'].isin(excludeRegions)]
        
    return groupTemplate  

def ReadUnformattedCSV(FPath):
    ElementsList = []
    with open(FPath, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file,delimiter=";")
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names in the CSV are {", ".join(row)}')
            line_count += 1
            print("Keys:",list(row.keys()))
            
            voxels_no = 1
            tmp={"id"          : int(row["id"]),
                  "pos_y" : int(float(row["pos_y"])),
                  "pos_x" : int(float(row["pos_x"])),
                  "pos_z" : int(float(row["pos_z"])),
                  "degree" : int(row["degree"]),
                  "isAtSampleBorder" : int(row["isAtSampleBorder"]),
                  "Blob_NumOfVoxels" : int(voxels_no)} 
            ElementsList.append(tmp)
    return ElementsList

def WriteCSV():
    with open(CSVFilePath.replace(".csv", "_Atlas.csv"), newline='', encoding='utf-8', mode='w') as csv_file:
        fieldnames = ['id', 'pos_x', 'pos_y', 'pos_z', 'degree', 'isAtSampleBorder', 'Blob_NumOfVoxels','Blob_AtlasLabel','Region_ColorHex','Region_Acronym']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for element in ElementsList:
            writer.writerow(element)


def AddAtlasLabels():
    for element in ElementsList:
        print("Processing ID " + str(element["id"]))
        LabelOfElement =  LabelImage[element["pos_x"], element["pos_y"], element["pos_z"]]
        element.update({"Blob_AtlasLabel":LabelOfElement})
        if np.abs(LabelOfElement) in ontologyDF["id"]:
            element.update({
                        "Region_ColorHex": ontologyDF[ontologyDF["id"]==np.abs(LabelOfElement)]["color-hex-triplet"].values[0],
                        "Region_Acronym": ontologyDF[ontologyDF["id"]==np.abs(LabelOfElement)]["acronym"].values[0]
                            })
    return ElementsList


def ReadFormattedCSV(FPath):        
    ElementsList = []
    with open(FPath, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names in the CSV are {", ".join(row)}')
            line_count += 1                
            ElementsList.append(row)
    return ElementsList


def AddRegionSizeToGroups():
    ClusteredList["RegionSizeVox"]    =np.nan
    ClusteredList["RegionSize (um3)"] =np.nan
    for i, row in ClusteredList.iterrows():
        print(datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + " starting label " + str(row["ColorGroup"]) + " for " + str( ontologyDF[ontologyDF["color-hex-triplet"]==row["ColorGroup"]]["id"].tolist() ))     
        RegionVoxelCount = 0
        tmp = ontologyDF[ontologyDF["color-hex-triplet"]==row["ColorGroup"]]["id"].tolist()
        for RegionLabel in tmp:
            print(datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + " proc.. " + str(RegionLabel))   
            RegionVoxelCount +=  np.sum( (LabelImage==RegionLabel)  | (LabelImage==RegionLabel*-1) )
        
        ClusteredList.at[i,'RegionSize']  = RegionVoxelCount
        ClusteredList["RegionSize (um3)"] = RegionVoxelCount * VoxelCorrectionFactor
    return ClusteredList


if __name__ == '__main__': 

    ontologyDF = parseOntologyXML(OntologyFilePath)

    LabelImage = np.asarray(nib.load(LabelFilePath).dataobj)
    
    ElementsList = ReadUnformattedCSV(CSVFilePath)
    ElementsList = AddAtlasLabels()
    WriteCSV()
        
    ClusteredList = collapseToColorGroup(ElementsList, excludeRegions=excludeRegionByColorHexList) 
    ClusteredList.to_csv(CSVFilePath.replace(".csv", "_AtlasGrouped.csv"), index=False)

    path_atlas_nodes = CSVFilePath.replace(".csv", "_Atlas.csv")
    path_atlas_groups = CSVFilePath.replace(".csv", "_AtlasGrouped.csv")
    path_atlas_encoded = CSVFilePath.replace(".csv", "_atlas_processed.csv")

    df_atlas_groups = pd.read_csv(path_atlas_groups, sep=',', usecols=['GroupAcronym', 'GroupedAcronyms'])
    df_atlas = pd.read_csv(path_atlas_nodes, sep=',', usecols=['Region_Acronym'])

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
        # print(str(df_atlas.at[k, 'Region_Acronym']))
        df_atlas.at[k, 'Region_Acronym'] = mapping[str(df_atlas.at[k, 'Region_Acronym'])]

    df_one_hot = pd.get_dummies(df_atlas, prefix=['Region_Acronym'], columns=['Region_Acronym'])  # assign column names
    df_one_hot.to_csv(path_atlas_encoded, sep=";", index=False)

    







































    
