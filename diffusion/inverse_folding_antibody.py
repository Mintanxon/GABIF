import pickle
import argparse
import sys
import os
import json
from tqdm import trange
from Bio import SeqIO, pairwise2
from Bio.Seq import Seq
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)
from dataset_src.generate_antibody_graph import prepare_graph,pdb2graph
from ema_pytorch import EMA
from gradeif_antibody import EGNN_NET,GraDe_IF

amino_acids_type = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                    'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

def align_index(aligned_seq):
    mapping = {}
    i, j = 0, 0
    while i < len(aligned_seq):
        if aligned_seq[i] != '-':
            mapping[j] = i
            j += 1
        i += 1
    return mapping

def get_seq_mapping(full_seq, pdb_seq):
    sequence1 = Seq(full_seq)
    sequence2 = Seq(pdb_seq)
    alignment = pairwise2.align.globalxs(sequence1, sequence2, -1, 0)[0]
    # print(pairwise2.format_alignment(*alignment))
    return align_index(alignment.seqB)

def graph_to_seq(x):
    return ''.join([amino_acids_type[i] for i in x.argmax(dim = 1).tolist()])

def get_gradeif_seq_by_chainid(x, chainid): 
    return graph_to_seq(x[input_graph.chainid == ord(chainid)])

def load_fasta_sequence(fasta_file):
    # only load first sequence
    return str(next(SeqIO.parse(fasta_file, 'fasta')).seq)

default_model_path = 'results/weight/BLOSUM_3M_small_antibody.pt'
redesigned_regions = ['CDRH1', 'CDRH2', 'CDRH3', 'CDRL1', 'CDRL2', 'CDRL3', 'AllCDRs', 'FullH', 'FullL', 'FullSequence']

parser = argparse.ArgumentParser(description='Graph Denoising Diffusion for Antibody Inverse Folding')
parser.add_argument('--ab_chainid1', default='H', help='Chain id for antibody heavy chain (default H)')
parser.add_argument('--ab_chainid2', default='L', help='Chain id for antibody light chain (default L)')
parser.add_argument('--ag_chainid', default='A', help='Chain id for antigen chain (default A)')
parser.add_argument('--proaffinity-inter-graph', help='Path for inter graph built by ProAffinity-GNN, only used when guidance')
parser.add_argument('--proaffinity-indi-graph', help='Path for individual graph built by ProAffinity-GNN, only used when guidance, +_1 represents graph1 and +_2 represents graph2')
parser.add_argument('--proaffinity-seq-ab1', help='Path for antibody heavy chain sequence fasta for ProAffinity-GNN prediction, only used when guidance, if not specified, use sequence from pdb')
parser.add_argument('--proaffinity-seq-ab2', help='Path for antibody light chain sequence fasta for ProAffinity-GNN prediction, only used when guidance, if not specified, use sequence from pdb')
parser.add_argument('--proaffinity-seq-ag', help='Path for antigen sequence for ProAffinity-GNN prediction, only used when guidance, if not specified, use sequence from pdb')
parser.add_argument('-i', '--input', required=True, help='Input antibody pdb, should be IMGT annotated')
parser.add_argument('-o', '--output', help='Output JSON file with generated sequences')
parser.add_argument('-m', '--model-path', default=default_model_path, help='GraDe-IF model path (default finetuned on SAbDab antibody dataset)')
parser.add_argument('-d', '--device', default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Which device for model to inference (default cuda:0)')
parser.add_argument('-n', '--num', default=10, type=int, help='Number of sequences to sample (default 10)')
parser.add_argument('-s', '--step', default=10, type=int, help='Number of denoising steps (default 10), fewer steps mean higher diversity while more steps mean higher recovery rate')
parser.add_argument('--alpha', default=200, type=float, help='Coefficient alpha used in guidance (default 200)')
parser.add_argument('--guidance', action="store_true", help='Whether to use affinity guidance by ProAffinity-GNN during GraDe-IF inference')
parser.add_argument('--redesigned-regions', nargs="+", choices=redesigned_regions, default=['CDRH3'], help='Which region should be redesigned (default CDRH3)')
args = parser.parse_args()

graph = pdb2graph(args.input,normalize_path = '../dataset_src/mean_attr.pt')
input_graph = Batch.from_data_list([prepare_graph(graph)])
ab_chainid1, ab_chainid2, ag_chainid = args.ab_chainid1, args.ab_chainid2, args.ag_chainid
device = args.device

if args.guidance:
    if not args.proaffinity_inter_graph or not args.proaffinity_indi_graph:
        parser.error('--guidance requires path for ProAffinity graphs')

    with open(args.proaffinity_inter_graph, 'rb') as f_graph1:
        graph = pickle.load(f_graph1)
        graph.edge_attr = graph.edge_attr.float()
        data = graph
        
    with open(args.proaffinity_indi_graph + '_1', 'rb') as f_graph2:
        graph = pickle.load(f_graph2)
        graph.edge_attr = graph.edge_attr.float()
        data1 = graph

    with open(args.proaffinity_indi_graph + '_2', 'rb') as f_graph3:
        graph = pickle.load(f_graph3)
        graph.edge_attr = graph.edge_attr.float()
        data2 = graph
    
    from antibody.proaffinity_gnn import predict_affinity, get_esm_embedding, get_esm_tokens, esm_to_gradeif_tokens
    wt_pKa = predict_affinity(data, data1, data2, device)
    target = wt_pKa + 2

cdrh1_index = input_graph.cdrh1[input_graph.chainid == ord(ab_chainid1)].nonzero().squeeze().tolist()
cdrh2_index = input_graph.cdrh2[input_graph.chainid == ord(ab_chainid1)].nonzero().squeeze().tolist()
cdrh3_index = input_graph.cdrh3[input_graph.chainid == ord(ab_chainid1)].nonzero().squeeze().tolist()
cdrl1_index = input_graph.cdrl1[input_graph.chainid == ord(ab_chainid2)].nonzero().squeeze().tolist()    
cdrl2_index = input_graph.cdrl2[input_graph.chainid == ord(ab_chainid2)].nonzero().squeeze().tolist()
cdrl3_index = input_graph.cdrl3[input_graph.chainid == ord(ab_chainid2)].nonzero().squeeze().tolist()

pdb_ab_seq1 = get_gradeif_seq_by_chainid(input_graph.x, ab_chainid1)
pdb_ab_seq2 = get_gradeif_seq_by_chainid(input_graph.x, ab_chainid2)
pdb_ag_seq = get_gradeif_seq_by_chainid(input_graph.x, ag_chainid)

antibody_seq1 = load_fasta_sequence(args.proaffinity_seq_ab1) if args.proaffinity_seq_ab1 else pdb_ab_seq1
antibody_seq2 = load_fasta_sequence(args.proaffinity_seq_ab2) if args.proaffinity_seq_ab2 else pdb_ab_seq2
antigen_seq = load_fasta_sequence(args.proaffinity_seq_ag) if args.proaffinity_seq_ag else pdb_ag_seq

gradeif_antibody_mapping1 = get_seq_mapping(antibody_seq1, pdb_ab_seq1)
gradeif_antibody_mapping2 = get_seq_mapping(antibody_seq2, pdb_ab_seq2)
gradeif_antigen_mapping = get_seq_mapping(antigen_seq, pdb_ag_seq)

redesigned_mask_symbol = '*'
redesigned_mask = torch.zeros_like(input_graph.cdr)
sequence_masks = {
    'CDRH1': input_graph.cdrh1, 
    'CDRH2': input_graph.cdrh2, 
    'CDRH3': input_graph.cdrh3, 
    'CDRL1': input_graph.cdrl1, 
    'CDRL2': input_graph.cdrl2, 
    'CDRL3': input_graph.cdrl3, 
    'AllCDRs': input_graph.cdr, 
    'FullH': input_graph.chainid == ord(ab_chainid1),
    'FullL': input_graph.chainid == ord(ab_chainid2),
    'FullSequence': (input_graph.chainid == ord(ab_chainid1)) | (input_graph.chainid == ord(ab_chainid2))
}

for region in args.redesigned_regions:
    redesigned_mask = redesigned_mask | sequence_masks[region]

redesigned_mask = redesigned_mask & sequence_masks['FullSequence']

results = {
    'heavy_chain': ab_chainid1,
    'light_chain': ab_chainid2,
    'antigen_chain': ag_chainid,
    'CDR_index': {
        'CDRH1': cdrh1_index,
        'CDRH2': cdrh2_index,
        'CDRH3': cdrh3_index,
        'CDRL1': cdrl1_index,
        'CDRL2': cdrl2_index,
        'CDRL3': cdrl3_index
    },
    'raw_sequence': {
        ab_chainid1: pdb_ab_seq1,
        ab_chainid2: pdb_ab_seq2,
        ag_chainid: pdb_ag_seq
    },
    'sampled_sequences': []
}

if args.guidance:
    results['wild_type_affinity'] = wt_pKa.item()
    def affinity_fn(sample_graph):
        _ab_seq1 = list(antibody_seq1)
        _ab_seq2 = list(antibody_seq2)
        sample_seq = graph_to_seq(sample_graph)
        masked_sample_seq = [redesigned_mask_symbol] * len(sample_seq)
        for i in redesigned_mask.nonzero():
            masked_sample_seq[i] = sample_seq[i]

        sample_ab_seq1 = ''.join(masked_sample_seq[i] for i in sequence_masks['FullH'].nonzero())
        for i in range(len(sample_ab_seq1)):
            if sample_ab_seq1[i] != redesigned_mask_symbol:
                _ab_seq1[gradeif_antibody_mapping1[i]] = sample_ab_seq1[i]
        
        sample_ab_seq2 = ''.join(masked_sample_seq[i] for i in sequence_masks['FullL'].nonzero())
        for i in range(len(sample_ab_seq2)):
            if sample_ab_seq2[i] != redesigned_mask_symbol:
                _ab_seq2[gradeif_antibody_mapping2[i]] = sample_ab_seq2[i]

        _ab_seq1 = ''.join(_ab_seq1)
        _ab_seq2 = ''.join(_ab_seq2)
        with torch.enable_grad():
            ab_token1, ab_token2 = get_esm_tokens(_ab_seq1), get_esm_tokens(_ab_seq2)
            ab_token_onehot1 = F.one_hot(ab_token1, num_classes=33).float().requires_grad_(True)
            ab_token_onehot2 = F.one_hot(ab_token2, num_classes=33).float().requires_grad_(True)

            data1.x = torch.cat((get_esm_embedding(ab_token1, ab_token_onehot1, device), get_esm_embedding(ab_token2, ab_token_onehot2, device)), 0)
            data.x = torch.cat((data1.x, data2.x.to(device)), 0)

            affinity = predict_affinity(data, data1, data2, device)
            loss = torch.nn.MSELoss()
            mse = loss(affinity, target)
            grad_x_ab1 = torch.autograd.grad(mse, ab_token_onehot1, retain_graph=True)[0][0,:,esm_to_gradeif_tokens].detach().requires_grad_(False)
            grad_x_ab2 = torch.autograd.grad(mse, ab_token_onehot2, retain_graph=True)[0][0,:,esm_to_gradeif_tokens].detach().requires_grad_(False)

        with torch.no_grad():
            grad_x = torch.zeros_like(sample_graph)
            i = 0
            for index in torch.where(sequence_masks['FullH'] & redesigned_mask)[0]:
                grad_x[index] = grad_x_ab1[gradeif_antibody_mapping1[i]]
                i += 1
            i = 0
            for index in torch.where(sequence_masks['FullL'] & redesigned_mask)[0]:
                grad_x[index] = grad_x_ab2[gradeif_antibody_mapping2[i]]
                i += 1

        affinity = affinity.item()
        del grad_x_ab1, grad_x_ab2, ab_token_onehot1, ab_token_onehot2, mse, data1.x, data.x
        torch.cuda.empty_cache()
        return affinity, grad_x

    input_graph.affinity_fn = affinity_fn

step = 500 // args.step
alpha = args.alpha if args.guidance else 0
ckpt = torch.load(args.model_path, map_location=device)
config = ckpt['config']
config['noise_type'] = 'uniform'
gnn = EGNN_NET(input_feat_dim=config['input_feat_dim'],hidden_channels=config['hidden_dim'],edge_attr_dim=config['edge_attr_dim'],dropout=config['drop_out'],n_layers=config['depth'],update_edge = config['update_edge'],embedding=config['embedding'],embedding_dim=config['embedding_dim'],embed_ss=config['embed_ss'],norm_feat=config['norm_feat'])
diffusion = GraDe_IF(model = gnn,config=config)
diffusion.alpha = alpha
diffusion = EMA(diffusion)
diffusion.load_state_dict(ckpt['ema'])

with torch.no_grad():
    for _ in trange(args.num, desc=f'Sample {args.redesigned_regions} from {os.path.split(args.input)[1]}'):
        prob, sample_graph = diffusion.ema_model.ddim_sample(input_graph,cond=~redesigned_mask,step=step)
        sample_seq = graph_to_seq(sample_graph)
        masked_sample_seq = [redesigned_mask_symbol] * len(sample_seq)
        for i in redesigned_mask.nonzero():
            masked_sample_seq[i] = sample_seq[i]
        masked_sample_ab_seq1 = ''.join(masked_sample_seq[i] for i in sequence_masks['FullH'].nonzero())
        masked_sample_ab_seq2 = ''.join(masked_sample_seq[i] for i in sequence_masks['FullL'].nonzero())

        sample_ab_seq1 = list(pdb_ab_seq1)
        for i in range(len(sample_ab_seq1)):
            if masked_sample_ab_seq1[i] != redesigned_mask_symbol:
                sample_ab_seq1[i] = masked_sample_ab_seq1[i]

        sample_ab_seq2 = list(pdb_ab_seq2)
        for i in range(len(sample_ab_seq2)):
            if masked_sample_ab_seq2[i] != redesigned_mask_symbol:
                sample_ab_seq2[i] = masked_sample_ab_seq2[i]

        sample_ab = {
            ab_chainid1: ''.join(sample_ab_seq1),
            ab_chainid2: ''.join(sample_ab_seq2),
            ag_chainid: pdb_ag_seq,
        }
        if args.guidance:
            sample_ab['affinity'] = affinity_fn(sample_graph)[0]
        results['sampled_sequences'].append(sample_ab)

output_file = args.output
if not args.output:
    folder, file = os.path.split(args.input)
    filename, _ = os.path.splitext(file)
    file = filename + '.json'
    output_file = os.path.join(folder, file)

with open(output_file, 'w') as f:
    json.dump(results, f, indent=1)