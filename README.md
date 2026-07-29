# GABIF: Graph Denoising Diffusion Model for Antibody Inverse Folding with Regressor Guidance

## Overview
<img width="2852" height="1796" alt="image" src="https://github.com/user-attachments/assets/3342d25f-b707-4318-961e-939a296646a5" />

## Requirements
To install requirements:
```
conda env create -f environment.yml
```
To run the guidance of ProAffinity-GNN, you need to prepare data as required by ProAffinity-GNN and ensure it run successfully. The instructions for ProAffinity-GNN can be found at: [https://github.com/legendzzy/ProAffinity-GNN](https://github.com/legendzzy/ProAffinity-GNN).

## Usage
`diffusion/inverse_folding_antibody.py` is the main entry of the model. 

Here is a brief usage example:
```shell
# for simply sample sequences without guidance
python inverse_folding_antibody.py \
    -i <input_imgt_antibody_pdb_path> \
    --ab_chainid1 H \
    --ab_chainid2 L \
    --ag_chainid A \
    --redesigned-regions CDRH3

# for sample sequences with ProAffinity-GNN guidance (Use 1ahw in ProAffinity-GNN demo data as example)
python inverse_folding_antibody.py \
    -i <input_imgt_antibody_pdb_path> \
    --ab_chainid1 A \
    --ab_chainid2 B \
    --ag_chainid C \
    --redesigned-regions CDRH3 \
    --proaffinity-inter-graph <ProAffinity-GNN_path>/graph/inter_graph/1ahw \
    --proaffinity-indi-graph <ProAffinity-GNN_path>/graph/individual_graph/1ahw \
    --proaffinity-seq-ab1 <ProAffinity-GNN_path>/FASTA/mixed/1AHW_1.fasta \
    --proaffinity-seq-ab2 <ProAffinity-GNN_path>/FASTA/mixed/1AHW_2.fasta \ 
    --proaffinity-seq-ag <ProAffinity-GNN_path>/FASTA/mixed/1AHW_3.fasta \
    --guidance
```
For more parameter details, please run `python inverse_folding_antibody.py -h`.

## Training and Experiment Details
Antibody data used in this work are all downloaded from https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab, training/val/test data are split by `antibody_split.pt`. We use default settings for finetuning and testing in our experiments, so we will not elaborate on the configuration settings. For more details, please refer to the codes of this repository (`diffusion/gradeif.py` for training and `diffusion/inverse_folding_antibody.py` for inference), [GraDe-IF](https://github.com/ykiiiiii/GraDe_IF) and [ProAffinity-GNN](https://github.com/legendzzy/ProAffinity-GNN). 

Training Settings
- Training Dataset: 3377 Sabdab antibody-antigen complexes for GABIF (with >700 affinity data for ProAffinity-GNN)
- Learning rate: 1e-4
- Weight decay: 1e-2
- Batch size: 64
- Training steps: 200k

Experiment Settings
- Input: 59 IMGT annotated antibody-antigen complex pdb
- Output: Mutant antibody sequences
- Sample num: 10 per CDR
- Denoising step size: 50
- Guidance scaling α: 300

