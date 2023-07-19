from utils.utils import get_mol, sanitize, get_motifs, get_fragment_mol
from utils.tu2smiles import to_smiles

import torch
from torch_geometric.data import Data
from rdkit.Chem import rdchem


ATOM = {
    'Mutagenicity': {
        6: 0,
        8: 1,
        17: 2,
        1: 3,
        7: 4,
        9: 5,
        35: 6,
        16: 7,
        15: 8,
        53: 9,
        11: 10,
        19: 11,
        3: 12,
        20: 13,
    },
}

EDGE = {
    'Mutagenicity': {
        rdchem.BondType.SINGLE: 0,
        rdchem.BondType.DOUBLE: 1,
        rdchem.BondType.TRIPLE: 2,
    },
}

def convert_data(data, motif_list, name):

    smiles = to_smiles(data, True, name)

    mol = get_mol(smiles)


    mol = sanitize(mol)
    motif_smiles, motifs, edges = get_motifs(mol)


    selected_atoms = []
    for i in range(len(motif_smiles)):
        if motif_smiles[i] in motif_list:
            selected_atoms += motifs[i]

    selected_atoms = sorted(list(set(selected_atoms)))


    new_mol = get_fragment_mol(mol, selected_atoms)

    atom_feature = [ATOM[name][atom.GetAtomicNum()] for atom in new_mol.GetAtoms()]

    new_x = torch.nn.functional.one_hot(torch.tensor(atom_feature), num_classes=14)
    new_edge_index = []
    new_edge_attr = []
    for bond in new_mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        new_edge_index.append((i, j))
        new_edge_index.append((j, i))
        new_edge_attr.append(EDGE[name][bond.GetBondType()])
        new_edge_attr.append(EDGE[name][bond.GetBondType()])
    new_edge_index = torch.tensor(new_edge_index, dtype=torch.long).t().contiguous()
    new_edge_attr = torch.tensor(new_edge_attr)
    new_edge_attr = torch.nn.functional.one_hot(new_edge_attr, num_classes=3)
    new_data = Data(x=new_x, edge_index=new_edge_index, edge_attr=new_edge_attr, y=data.y)
    sparsity = 1 - len(selected_atoms)/data.num_nodes

    return new_data, sparsity