# GraDe_abIF: Graph Denoising Diffusion Model for Antibody Inverse Folding with Regressor Guidance

## Requirements
To install requirements:
```
conda env create -f environment.yml
```
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