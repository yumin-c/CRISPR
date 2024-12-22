import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset
from typing import Tuple
from tqdm import tqdm
import random


def preprocess_seq(data):
    print("Start preprocessing the sequence done 2d")
    length = 74

    seq_onehot = np.zeros((len(data), 1, length, 4), dtype=float)
    print(np.shape(data), len(data), length)
    for l in tqdm(range(len(data))):
        for i in range(length):

            try:
                data[l][i]
            except Exception:
                print(data[l], i, length, len(data))

            if data[l][i] in "Aa":
                seq_onehot[l, 0, i, 0] = 1
            elif data[l][i] in "Cc":
                seq_onehot[l, 0, i, 1] = 1
            elif data[l][i] in "Gg":
                seq_onehot[l, 0, i, 2] = 1
            elif data[l][i] in "Tt":
                seq_onehot[l, 0, i, 3] = 1
            elif data[l][i] in "Xx":
                pass
            else:
                print("Non-ATGC character " + data[l])
                print(i)
                print(data[l][i])
                sys.exit()

    print("Preprocessed the sequence")
    return seq_onehot


def seq_concat(data, col1='WT74_On', col2='Edited74_On'):
    wt = preprocess_seq(data[col1])
    ed = preprocess_seq(data[col2])
    g = np.concatenate((wt, ed), axis=1)

    return 2 * g - 1


def select_cols(data):
    features = data.loc[:, ['PBSlen', 'RTlen', 'RT-PBSlen', 'Edit_pos', 'Edit_len', 'RHA_len', 'type_sub',
                            'type_ins', 'type_del', 'Tm1', 'Tm2', 'Tm2new', 'Tm3', 'Tm4', 'TmD',
                            'nGCcnt1', 'nGCcnt2', 'nGCcnt3', 'fGCcont1', 'fGCcont2', 'fGCcont3', 'MFE3', 'MFE4', 'DeepSpCas9_score']]
    if 'Measured_PE_efficiency' in data.columns:
        target = data['Measured_PE_efficiency']
    elif 'real_Off-target_%' in data.columns:
        target = data['real_Off-target_%']
    else:
        target = None
        
    return features, target


class GeneFeatureDataset(Dataset):

    def __init__(
        self,
        gene: torch.Tensor = None,
        features: torch.Tensor = None,
        target: torch.Tensor = None,
        fold: int = None,
        mode: str = 'train',
        fold_list: np.ndarray = None,
        offtarget_mutate: float = 0.,
        ontarget_mutate: float = 0.,
        subsampling: int = -1,
        random_seed: int = 0,
    ):
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

        self.fold = fold
        self.mode = mode
        self.fold_list = fold_list
        self.offtarget_mutate = offtarget_mutate
        self.ontarget_mutate = ontarget_mutate
        self.subsampling = subsampling
        self.atgc = torch.tensor([
            [1., -1., -1., -1.],
            [-1., 1., -1., -1.],
            [-1., -1., 1., -1.],
            [-1., -1., -1., 1.]], dtype=torch.float32, device=gene.device)

        if self.fold_list is not None:
            self.indices = self._select_fold()
            self.gene = gene[self.indices]
            self.features = features[self.indices]
            self.target = target[self.indices]
        else:
            self.gene = gene
            self.features = features
            self.target = target

    def _select_fold(self):
        selected_indices = []

        if self.mode == 'valid':  # Select a single group
            for i in range(len(self.fold_list)):
                if self.fold_list[i] == self.fold:
                    selected_indices.append(i)
        elif self.mode == 'train':  # Select others
            for i in range(len(self.fold_list)):
                if self.fold_list[i] != self.fold and self.fold_list[i] != 'Test':
                    selected_indices.append(i)
        elif self.mode == 'finalizing':
            for i in range(len(self.fold_list)):
                selected_indices.append(i)

        if self.subsampling > 0:
            selected_indices = random.sample(selected_indices, self.subsampling)

        return selected_indices
        

    def __len__(self):
        return len(self.gene)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        gene = self.gene[idx]
        features = self.features[idx]
        target = self.target[idx]
        
        if self.offtarget_mutate and self.offtarget_mutate > torch.rand(1): # Transform part of data to dummy data with off-target efficiency of 0%
            o_indices = gene[1, :, :].sum(dim=1) != -4.
            proportion = 0.4
            mutate_indices = o_indices & (torch.rand(74, device=o_indices.device) < proportion)
            gene[0, mutate_indices] = self.atgc[torch.randint(4, (mutate_indices.sum().cpu().numpy().item(),))]
            target = 1e-4 * torch.ones_like(target)

        if self.ontarget_mutate and self.ontarget_mutate > torch.rand(1): # Mutate nucleotides in non-interactive region of a target DNA
            x_indices = gene[1, :, :].sum(dim=1) == -4.
            proportion = 0.2
            mutate_indices = x_indices & (torch.rand(74, device=x_indices.device) < proportion)
            gene[0, mutate_indices] = self.atgc[torch.randint(4, (mutate_indices.sum().cpu().numpy().item(),))]

        return gene, features, target
    

class MultiPEDataset(Dataset):

    def __init__(
        self,
        csv_list: list = None,
        fold: int = None,
        mode: str = 'train',
        scaler = None,
        offtarget_mutate: float = 0.,
        ontarget_mutate: float = 0.,
        random_seed: int = 216,
    ):
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

        self.csv_list = csv_list
        self.fold = fold
        self.mode = mode
        self.scaler = scaler
        self.offtarget_mutate = offtarget_mutate
        self.ontarget_mutate = ontarget_mutate
        self.atgc = torch.tensor([
            [1., -1., -1., -1.],
            [-1., 1., -1., -1.],
            [-1., -1., 1., -1.],
            [-1., -1., -1., 1.]], dtype=torch.float32)
        
        self.gene, self.features, self.target = self._process_and_merge_files()
        
    def _process_and_merge_files(self):
        merged_gene = None
        merged_features = None
        merged_target = None
        
        for csv_path in self.csv_list:
            gene, features, target = self._process_file(csv_path)
            if merged_gene is None:
                merged_gene = gene
                merged_features = features
                merged_target = target
            else:
                merged_gene = np.concatenate((merged_gene, gene), axis=0)
                merged_features = np.concatenate((merged_features, features), axis=0)
                merged_target = np.concatenate((merged_target, target), axis=0)
        
        if self.scaler is None:
            scaler = StandardScaler()
            features = scaler.fit_transform(features)
            self.scaler = scaler
        else:
            features = self.scaler.transform(features)
                
        merged_gene = torch.tensor(merged_gene, dtype=torch.float32)
        merged_features = torch.tensor(merged_features, dtype=torch.float32)
        merged_target = torch.tensor(merged_target, dtype=torch.float32)
        
        return merged_gene, merged_features, merged_target
        
    def _process_file(self, csv_path):
        data = pd.read_csv(csv_path)
        base_dir = os.path.dirname(csv_path)
        genes_dir = os.path.join(base_dir, "genes")
        os.makedirs(genes_dir, exist_ok=True)
        file_name = os.path.basename(csv_path).replace(".csv", ".npy")
        gene_path = os.path.join(genes_dir, file_name)
        
        if not os.path.isfile(gene_path):
            gene = seq_concat(data)
            np.save(gene_path, gene)
        else:
            gene = np.load(gene_path)
        
        features = data.loc[:, ['PBSlen', 'RTlen', 'Edit_pos', 'Edit_len', 'RHA_len', 'type_sub', 'type_ins', 'type_del',
                                'Tm1', 'Tm2', 'Tm2new', 'Tm3', 'Tm4', 'TmD', 'MFE3', 'MFE4']] # rm nGC, fGC, PBS+RTlen
        target = data.loc[:, ['Measured_PE_efficiency', 'type_sub', 'type_ins', 'type_del']]
        
        dataset_type_encoding = self._encode_dataset_type(csv_path)
        cas9_score = self._get_cas9_score(data, dataset_type_encoding['nrch'])
        
        for key, value in dataset_type_encoding.items():
            features[key] = value
        features['cas9_score'] = cas9_score
    
        features = features.to_numpy()
        target = target.to_numpy()
        
        if self.fold is not None:
            fold_list = data['Fold']
            indices = self._select_fold(fold_list)
            return gene[indices], features[indices], target[indices]
        else:
            return gene, features, target

    def _get_cas9_score(self, data, cas9_type):
        if cas9_type == 1: # nrch
            return data.loc[:, 'SpCas9-NRCH']
        else:
            return data.loc[:, 'SpCas9-SpCas9']
    
    def _encode_dataset_type(self, csv_path):
        dataset_feature_list = [
            'nicking_sgrna', 'mlh1dn', 'maxcas9', 'maxrt', 'la', 'epegrna', 'nrch'
        ]
        dataset_type = dict(keys=dataset_feature_list)
        
        if 'PE2' in csv_path:
            dataset_type = {
                'nicking_sgrna': 0,
                'mlh1dn': 0,
                'maxcas9': 0,
                'maxrt': 0,
                'la': 0,
            }
        elif 'PE3' in csv_path:
            dataset_type = {
                'nicking_sgrna': 1,
                'mlh1dn': 0,
                'maxcas9': 0,
                'maxrt': 0,
                'la': 0,
            }
        elif 'PE4' in csv_path:
            dataset_type = {
                'nicking_sgrna': 0,
                'mlh1dn': 1,
                'maxcas9': 0,
                'maxrt': 0,
                'la': 0,
            }
        elif 'PE5' in csv_path:
            dataset_type = {
                'nicking_sgrna': 1,
                'mlh1dn': 1,
                'maxcas9': 0,
                'maxrt': 0,
                'la': 0,
            }
        elif 'PE7' in csv_path:
            dataset_type = {
                'nicking_sgrna': 0,
                'mlh1dn': 0,
                'maxcas9': 1,
                'maxrt': 1,
                'la': 1,
            }
        elif 'PE4+7' in csv_path:
            dataset_type = {
                'nicking_sgrna': 0,
                'mlh1dn': 1,
                'maxcas9': 1,
                'maxrt': 1,
                'la': 1,
            }
            
        
        if 'max' in csv_path:
            dataset_type['maxcas9'] = 1
            dataset_type['maxrt'] = 1
        
        if 'epegRNA' in csv_path:
            dataset_type['epegrna'] = 1
        else:
            dataset_type['epegrna'] = 0
        
        if 'NRCH' in csv_path:
            dataset_type['nrch'] = 1 # consider removal
        else:
            dataset_type['nrch'] = 0
        
        return dataset_type

    def _select_fold(self, fold_list):
        selected_indices = []

        if self.mode == 'valid':  # Select a single group
            for i in range(len(fold_list)):
                if fold_list[i] == self.fold:
                    selected_indices.append(i)
        elif self.mode == 'train':  # Select others
            for i in range(len(fold_list)):
                if fold_list[i] != self.fold and fold_list[i] != 'Test':
                    selected_indices.append(i)
        elif self.mode == 'all':
            for i in range(len(fold_list)):
                selected_indices.append(i)

        return selected_indices
        
    def __len__(self):
        return len(self.gene)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        gene = self.gene[idx]
        features = self.features[idx]
        target = self.target[idx]
        
        if self.offtarget_mutate and self.offtarget_mutate > torch.rand(1): # Transform part of data to dummy data with off-target efficiency of 0%
            o_indices = gene[1, :, :].sum(dim=1) != -4.
            proportion = 0.4
            mutate_indices = o_indices & (torch.rand(74, device=o_indices.device) < proportion)
            gene[0, mutate_indices] = self.atgc[torch.randint(4, (mutate_indices.sum().cpu().numpy().item(),))]
            target = 1e-4 * torch.ones_like(target)

        if self.ontarget_mutate and self.ontarget_mutate > torch.rand(1): # Mutate nucleotides in non-interactive region of a target DNA
            x_indices = gene[1, :, :].sum(dim=1) == -4.
            proportion = 0.2
            mutate_indices = x_indices & (torch.rand(74, device=x_indices.device) < proportion)
            gene[0, mutate_indices] = self.atgc[torch.randint(4, (mutate_indices.sum().cpu().numpy().item(),))]

        return gene, features, target