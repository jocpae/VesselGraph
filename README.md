## Introduction
![alt text](synthetic_overview.png "Logo Title Text 1")

Welcome to the project page of *DeepVesselGraph*  A Dataset and Benchmark for Graph Learning and Neuroscience. <br/>

Biological neural networks define human and mammalian brain function and intelligence and form ultra-large, spatial, structured graphs. Their neuronal organization is closely interconnected with the spatial organization of the brain's microvasculature, which supplies oxygen to the neurons and builds a complementary spatial graph. In this project we are providing an extendable dataset of whole-brain vessel graphs based on various multi-center imaging protocols. 

This new dataset paves a pathway towards translating advanced graph learning research into the field of neuroscience. Complementarily, the new dataset raises challenging graph learning research questions for the machine learning community, for example how to incorporate biological priors in a meaningful and interpretable way into learning algorithms. 

## Table of contents

* [Datasets](#dataset-instruction)
* [Baselines](#baseline-instruction)
* [Biological Context](#bio)


## Dataset Instruction
#### 1. Generate Raw Graph fron Segmentation using Voreen
Use [Voreen Graph Generation Tool](https://github.com/jqmcginnis/voreen) to make the `node_list` and `edge_list` from a segmentation volume.

#### 2. Preprocess Dataset

Go to `./source/dataset_preprocessing/` and run `process_edge_list.py` with arguments of `--node_list` and `--edge_list`

#### 3. Generate Atlas features
Got to `./source/feature_generation/atlas_annotation/` and run `generate_node_atlas_labels.py` with arguments of `--node_list` and `--edge_list`
#### 4. Convert to Pytorch-Geometric Data
Got to `./source/pytorch_dataset/` and run `link_dataset.py` and `node_dataset.py` to create pytorch-geometric compatible dataset for link-rediction and node-classification task.
#### 5. Convert to OGB compatible format
Got to `./source/ogb_dataset/link_prediction/` and run `update_ogbl_master.sh` for link-rediction task

Go to `./source/ogb_dataset/node_classification/` and run `update_ogbn_master.sh` for node-classification task.
## Baseline Instruction

## Contribute 

We are a living and continously maintained repository! Therefore, we welcome contributions of additional datasets and methods! There is multiple ways to contribute; if you are willing to share whole brain segmentations and graphs .... 


## Reference  

 R. J. L. Townshend, M. Vögele, P. Suriana, A. Derry, A. Powers, Y. Laloudakis, S. Balachandar, B. Jing, B. Anderson, S. Eismann, R. Kondor, R. B. Altman, R. O. Dror "ATOM3D: Tasks On Molecules in Three Dimensions", [arXiv:2012.04035](https://arxiv.org/abs/2012.04035)
  
Please cite this work if some of the ATOM3D code or datasets are helpful in your scientific endeavours. For specific datasets, please also cite the respective original source(s), given in the preprint.



## License 

Our project is licensed under the [MIT license](https://github.com/jocpae/VesselGraph/LICENSE).
