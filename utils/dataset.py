from torch_geometric.datasets import TUDataset
from torch_geometric.data import InMemoryDataset, Data
import os
import urllib.request
import torch
from torch.utils.data import Dataset
from utils.tu2smiles import to_smiles, convert_data
from utils.utils import get_mol, get_motifs, sanitize_smiles
from utils.motif_dataset import MotifDataset
import json
import pandas as pd
import csv
from tqdm import tqdm
from rdkit import Chem
import numpy as np
import json
from torch_geometric.datasets import MoleculeNet
import os

###### To-Do: Make it create new vocabulary based on smiles representation.

class HeterTUDataset(InMemoryDataset):
    # def __init__(self, root, name) -> object:
    def __init__(self, root, name, num_nodes, transform=None, pre_transform=None, pre_filter=None):
        # super().__init__(root, transform, pre_transform, pre_filter)
        self.name = name
        self.smiles_list = []
        self.num_nodes = num_nodes
        self.motif_vocab = {}
        self.cliques_edge = {}
        self.check = {}
        if self.name in ["PTC_MR", "Mutagenicity", "COX2_MD", "COX2", "BZR", "BZR_MD", "DHFR_MD", "MUTAG", "NCI1", "NCI109", "ER_MD", "PTC_FR", "PTC_MM", "PTC_FM"]:
            self.dataset = TUDataset("data/", self.name)
            self.data_type = "TUData"
        else:
            smiles_list = []
            self.raw_dataset = MoleculeNet('ori_data/', self.name)
            labels = []
            for data in self.raw_dataset:
                smiles_list.append(data.smiles)
                # if data.y.item() == 0:
                #     print('hh')
                # print(data.y.squeeze().tolist())
                labels.append(data.y.squeeze().tolist())
            labels = pd.DataFrame(labels)
            labels = labels.replace(0, -1)
            labels = labels.fillna(0).values.tolist()
            if len(self.raw_dataset) != len(smiles_list):
                print('Wrong raw data mapping!')
            self.dataset = tuple([smiles_list, labels])
            self.data_type = "MolNet"
        self.vocab_id = {}
        self.inv_vocab_mapping = {}

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def processed_file_names(self):
        return ['heter_data.pt']
    
    @property
    def raw_file_names(self):
        return ["smiles.csv"]

    def add_id(self, mol):
        for atom in mol.GetAtoms():
            atom.SetIntProp("OriID", atom.GetIdx())
            # print(atom.GetIntProp("OriID"))
        return mol
    
    def get_bonds(self, mol):
        bonds_list = []
        for bond in mol.GetBonds():
            b1 = bond.GetBeginAtomIdx()
            b2 = bond.GetEndAtomIdx()
            bonds_list.append(sorted(tuple([b1, b2])))
            # bonds_list.append(tuple([b2, b1]))
        return bonds_list
    
    def generate_heter(self, smiles, count):
        mol = get_mol(smiles)

        mol = self.add_id(mol)
        n_atoms = mol.GetNumAtoms()
        # print(f"Number of atoms: {mol.GetNumAtoms()}")
        bonds_list = self.get_bonds(mol)
        node_feature = torch.zeros((1,self.num_nodes))

        motifs_smiles, motifs, edges = get_motifs(mol)
        for i in range(len(motifs_smiles)):
            if motifs_smiles[i] not in self.motif_vocab:
                self.motif_vocab[motifs_smiles[i]] = len(self.motif_vocab)
            node_feature[0, self.motif_vocab[motifs_smiles[i]]] = 1
            edge = torch.tensor([[self.num_nodes+count, self.motif_vocab[motifs_smiles[i]]], [self.motif_vocab[motifs_smiles[i]], self.num_nodes+count]])
            self.heter_edge_index = torch.cat((self.heter_edge_index, edge), dim=1)
        self.heter_node_features = torch.cat((self.heter_node_features, node_feature), dim=0)
        for item in edges:
            edge = torch.tensor([[self.motif_vocab[motifs_smiles[item[0]]], self.motif_vocab[motifs_smiles[item[1]]]], [self.motif_vocab[motifs_smiles[item[1]]], self.motif_vocab[motifs_smiles[item[0]]]]])
            self.heter_edge_index = torch.cat((self.heter_edge_index, edge), dim=1)
            
        return 0
    
    def process(self):
        data_list = []
        count = 0
        
        self.heter_node_features = torch.eye(self.num_nodes)
        self.heter_edge_index = torch.empty((2, 0), dtype=torch.long)
        if self.name == "clintox":
            self.labels = torch.empty((self.num_nodes, 2), dtype=torch.long)
        elif self.name == "sider":
            self.labels = torch.empty((self.num_nodes, 27), dtype=torch.long)
        elif self.name == "tox21":
            self.labels = torch.empty((self.num_nodes, 12), dtype=torch.long)
        elif self.name == "toxcast":
            self.labels = torch.empty((self.num_nodes, 617), dtype=torch.long)
        elif self.name == "muv":
            self.labels = torch.empty((self.num_nodes, 17), dtype=torch.long)
        else:
            self.labels = torch.empty((self.num_nodes, 1), dtype=torch.long)
        heter_edge_attr = torch.empty((0,))
        if self.data_type == "TUData":
            selected_idx = []
            print(len(self.dataset))
            for i, data in enumerate(self.dataset):
                smiles = to_smiles(data, True, self.name)
                smiles = sanitize_smiles(smiles)
                if smiles == None:
                    continue
                else:
                    # print(f"smiles: {smiles}")
                    label = data.y
                    if label.item() == -1:
                        label = torch.tensor([0])
                    self.labels = torch.cat((self.labels.squeeze(), label), dim=0)
                    # new_data = convert_data(self.name, data)
                    # data_list.append(new_data)
                    selected_idx.append(i)
                    self.smiles_list.append(smiles)
                    self.generate_heter(smiles, count)
                    count += 1
           
            # motif_dataset = MotifDataset(sorted(self.new_vocab, key=lambda k: self.new_vocab[k]), "motif_data/"+self.name)
        elif self.data_type == "MolNet":
            count_selected = 0
            selected_idx = []
            for i, (smiles, label) in tqdm(enumerate(zip(*self.dataset))):
                smiles = sanitize_smiles(smiles)
                if smiles is None:
                    continue
                else:
                    selected_idx.append(i)
                    label = torch.tensor([label])
                    # data_list.append(self.raw_dataset[i])
                    if len(label.shape) == 1:
                        label.unsqueeze(dim=0)
                    self.labels = torch.cat((self.labels, label), dim=0)
                    self.smiles_list.append(smiles)
                    self.generate_heter(smiles, count)
                    count += 1
        # with open("checkpoints/"+self.name+"_motifvocab.txt", 'w') as convert_file:
        #     convert_file.write(json.dumps(self.motif_vocab))
        current_path = os.getcwd()
        if not os.path.exists(current_path + "/motif_data/"+self.name+"/raw/"):
            os.makedirs(current_path + "/motif_data/"+self.name+"/raw/")
        torch.save(sorted(self.motif_vocab, key=lambda k: self.motif_vocab[k]), current_path + "/motif_data/"+self.name+"/raw/raw_motif_data.pt")
        # if not os.path.exists(current_path + "/raw_data/"+self.name+"/raw/"):
        #     os.makedirs(current_path + "/raw_data/"+self.name+"/raw/")
        # torch.save(data_list, current_path + "/raw_data/"+self.name+"/raw/raw_data.pt")
        self.heter_edge_index = torch.unique(self.heter_edge_index, dim=1)

        # Add node attributes
        node_id = [x for x in range(self.num_nodes+count)]
        node_attr = torch.tensor(node_id)
        n = max(self.motif_vocab.values()) + 1
        motif_list = [0]*n
        for key, value in self.motif_vocab.items():
            motif_list[value] = key
        smiles_list = motif_list + self.smiles_list

        if self.data_type == "MolNet":
            heter_data = Data(x=self.heter_node_features, edge_index=self.heter_edge_index, y=self.labels, smiles_list=smiles_list, selected_idx=selected_idx)
        elif self.data_type == "TUData":
            heter_data = Data(x=self.heter_node_features, edge_index=self.heter_edge_index, y=self.labels, node_attr=node_attr, smiles_list=smiles_list, selected_idx=selected_idx)
        print(f"Number of motifs: {len(self.motif_vocab)}")
        print(self.processed_paths)
        print(f"labels: {heter_data.y.size()}")
        print(f"Number of edges: {heter_data.edge_index.size()}")
        print(f"Number of graphs: {count}")

        data_smiles_series = pd.Series(self.smiles_list)
        data_smiles_series.to_csv(os.path.join(self.processed_dir,
                                               'smiles.csv'), index=False,
                                  header=False)

        print(heter_data)
        # print(stop)

        torch.save(self.collate([heter_data]), self.processed_paths[0])