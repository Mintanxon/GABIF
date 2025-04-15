      
import os
import argparse
import pandas as pd
from dataset_src.antibody_parser import Cath_imem,dataset_argument
from torch.optim import Adam
from torch_geometric.data import Batch,Data
from dataset_src.utils import NormalizeProtein
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
import torch.nn.functional as F
import torch
from tqdm import tqdm

amino_acids_type = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

def get_struc2ndRes(pdb_filename):
    struc_2nds_res_alphabet = ['E', 'L', 'I', 'T', 'H', 'B', 'G', 'S']
    char_to_int = dict((c, i) for i, c in enumerate(struc_2nds_res_alphabet))

    p = PDBParser()
    structure = p.get_structure('random_id', pdb_filename)
    model = structure[0]
    dssp = DSSP(model, pdb_filename, dssp='mkdssp')

    # From model, extract the list of amino acids
    model_residues = [(chain.id, residue.id[1]) for chain in model for residue in chain if residue.id[0] == ' ']
    # From DSSP, extract the list of amino acids
    dssp_residues = [(k[0], k[1][1]) for k in dssp.keys()]

    # Determine the missing amino acids
    missing_residues = set(model_residues) - set(dssp_residues)

    # Initialize a list of integers for known secondary structures,
    # and another list of zeroes for one-hot encoding
    integer_encoded = []
    one_hot_list = torch.zeros(len(model_residues), len(struc_2nds_res_alphabet))

    current_position = 0
    for chain_id, residue_num in model_residues:
        dssp_key = (chain_id, (' ', residue_num, ' '))
        if (chain_id, residue_num) not in missing_residues and dssp_key in dssp:
            
            sec_structure_char = dssp[dssp_key][2]
            sec_structure_char = sec_structure_char.replace('-', 'L')
            integer_encoded.append(char_to_int[sec_structure_char])

            one_hot = F.one_hot(torch.tensor(integer_encoded[-1]), num_classes=8)
            one_hot_list[current_position] = one_hot
        else:
            # print(pdb_filename,'Missing residue: ', chain_id, residue_num, 'fill with 0')
            pass
        current_position += 1
    ss_encoding = one_hot_list[:current_position]
    return ss_encoding

def prepare_graph(data):
    del data['distances']
    del data['edge_dist']
    mu_r_norm=data.mu_r_norm

    extra_x_feature = torch.cat([data.x[:,20:],mu_r_norm],dim=1)
    try:
        re_norm = torch.ones(1)*data.re_norm
        label_mask = torch.ones(1, dtype=torch.bool)
    except AttributeError:
        re_norm = torch.zeros(1)
        label_mask = torch.zeros(1, dtype=torch.bool)
    graph = Data(
        x=data.x[:, :20],
        extra_x = extra_x_feature,
        pos=data.pos,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
        ss = data.ss[:data.x.shape[0],:],
        sasa = data.x[:,20],
        re_norm = re_norm,
        label_mask = label_mask,
        chainid = data.chainid,
        mu_r_norm = data.mu_r_norm,
        cdr = data.cdr,
        cdrh1 = data.cdrh1,
        cdrh2 = data.cdrh2,
        cdrh3 = data.cdrh3,
        cdrl1 = data.cdrl1,
        cdrl2 = data.cdrl2,
        cdrl3 = data.cdrl3,
        isab = data.isab
    )
    return graph

def pdb2graph(filename,normalize_path = 'dataset_src/mean_attr.pt',chain_id=None):
    #### dataset  ####
    dataset_arg = dataset_argument(n=51)
    dataset = Cath_imem(dataset_arg['root'], dataset_arg['name'], split='test',
                                divide_num=dataset_arg['divide_num'], divide_idx=dataset_arg['divide_idx'],
                                c_alpha_max_neighbors=dataset_arg['c_alpha_max_neighbors'],
                                set_length=dataset_arg['set_length'],
                                struc_2nds_res_path = dataset_arg['struc_2nds_res_path'],
                                random_sampling=True,diffusion=True)
    rec, rec_coords, c_alpha_coords, n_coords, c_coords, cdr , CDRH1,CDRH2,CDRH3,CDRL1,CDRL2,CDRL3,IS_AB,CHAIN_ID = dataset.get_receptor_inference(filename,ID=chain_id)
    struc_2nd_res = get_struc2ndRes(filename)
    rec_graph = dataset.get_calpha_graph(
                rec, c_alpha_coords, n_coords, c_coords, rec_coords, cdr, CDRH1,CDRH2,CDRH3,CDRL1,CDRL2,CDRL3,IS_AB,CHAIN_ID,struc_2nd_res)
    if rec_graph:
        normalize_transform = NormalizeProtein(filename=normalize_path)
        
        graph = normalize_transform(rec_graph)
        return graph
    else:
        return None



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_dir', type = str,default='dataset/CD8_antibody/pdb/',
                        help='pdb dir')
    parser.add_argument('--save_dir', type = str,default='dataset/CD8_antibody/process_new/',help='save dir')
    parser.add_argument('--label',action='store_true',help='whether update edge feature in egnn')
    args = parser.parse_args()

    pdb_dir = args.pdb_dir
    save_dir = args.save_dir


    pdb_list = os.listdir(pdb_dir)

    for pdb_id in tqdm(pdb_list):
        pdb_path = os.path.join(pdb_dir,pdb_id)
        graph = pdb2graph(pdb_path,'dataset_src/mean_attr.pt')
        pdb_code = pdb_id.replace('.pdb','')
        try:
            graph = pdb2graph(pdb_path,'dataset_src/mean_attr.pt')
            pdb_code = pdb_id.replace('.pdb','')
            graph.pdb_code = pdb_code
            torch.save(graph, f'{save_dir}/{pdb_code}.pt')
        except Exception as e:
            print(f'error with {pdb_id}: {e}')
    # generate a batch of protein graph
    # save_dir = 'dataset/antibody/process_with_CDRH1/'
    # if not os.path.exists(save_dir+'/train'):
    #     os.makedirs(save_dir+'/train')
    #     os.makedirs(save_dir+'/val')
    #     os.makedirs(save_dir+'/test') 


    # split = torch.load('dataset/antibody/antibody_split.pt')
    # train_idx = split['train']
    # val_idx = split['valid']
    # test_idx = split['test']

    # for pdb_id in tqdm(test_idx):
    #     pdb_path = 'dataset/antibody/all_structures/imgt/'+pdb_id+'.pdb'
    #     try:
    #         graph = pdb2graph(pdb_path,'dataset_src/mean_attr.pt')
    #         torch.save(graph, f'{save_dir}/test/{pdb_id}.pt')
    #     except Exception as e:
    #         print(f'error with {pdb_id}: {e}')


    # for pdb_id in tqdm(train_idx):
    #     pdb_path = 'dataset/antibody/all_structures/imgt/'+pdb_id+'.pdb'
    #     try:
    #         graph = pdb2graph(pdb_path,'dataset_src/mean_attr.pt')
    #         torch.save(graph, f'{save_dir}/train/{pdb_id}.pt')
    #     except Exception as e:
    #         print(f'error with {pdb_id}: {e}')

    # for pdb_id in tqdm(val_idx):
    #     pdb_path = 'dataset/antibody/all_structures/imgt/'+pdb_id+'.pdb'
    #     try:
    #         graph = pdb2graph(pdb_path,'dataset_src/mean_attr.pt')
    #         torch.save(graph, f'{save_dir}/val/{pdb_id}.pt')
    #     except Exception as e:
    #         print(f'error with {pdb_id}: {e}')