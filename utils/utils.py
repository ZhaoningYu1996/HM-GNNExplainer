import rdkit.Chem as Chem
from collections import defaultdict

def clean_data(smile):
    mol = get_mol(smile)
    if mol == None:
        return False
    else:
        return True

def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return None
    Chem.Kekulize(mol, clearAromaticFlags=True) # Add clearAromaticFlags to avoid error
    if mol is None:
        return None
    return mol

def get_smiles(mol):
    return Chem.MolToSmiles(mol, kekuleSmiles=True)

def sanitize(mol):
    try:
        smiles = get_smiles(mol)
        mol = get_mol(smiles)
    except Exception as e:
        return None
    return mol

def sanitize_smiles(smiles):
    try:
        mol = get_mol(smiles)
        smiles = get_smiles(mol)
    except Exception as e:
        return None
    
    return smiles

def copy_atom(atom):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom

def copy_edit_mol(mol):
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)
        new_mol.AddAtom(new_atom)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
    return new_mol

def get_fragment_mol(mol, atom_indices):
    edit_mol = Chem.EditableMol(Chem.Mol())
    for idx in atom_indices:
        edit_mol.AddAtom(mol.GetAtomWithIdx(idx))
    for i, idx1 in enumerate(atom_indices):
        for idx2 in atom_indices[i + 1:]:
            bond = mol.GetBondBetweenAtoms(idx1, idx2)
            if bond is not None:
                edit_mol.AddBond(atom_indices.index(idx1), atom_indices.index(idx2), order=bond.GetBondType())

    submol = edit_mol.GetMol()
    # submol = sanitize(submol)

    return submol

def get_clique_smile(mol, atoms):
    smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=False)
    return sanitize_smiles(smiles)

def get_cliques(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        atom_list = []
        for atom in mol.GetAtoms():
            atom_list.append(atom.GetIntProp("OriID"))
        if len(atom_list) == 1:
            return [atom_list], [[0]], [], 1
        else:
            print("Wrong atom list!!! Error!!!")

    cliques = []
    cliques_ori = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            cliques.append([a1,a2])

    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
    for ring in ssr:
        for id in ring:
            atom = mol.GetAtomWithIdx(id)
    cliques.extend(ssr)
    cliques = [c for c in cliques if len(c) > 0]
    nei_list = [[] for i in range(n_atoms)]
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)

    return cliques, nei_list, n_atoms

def get_cliques_edges(mol, cliques, nei_list, n_atoms):
    if len(cliques) == 1:
        return [], cliques

    #Build edges and add singleton cliques
    edges = defaultdict(int)
    for atom in range(n_atoms):
        if len(nei_list[atom]) <= 1: 
            continue
        cnei = nei_list[atom]
        bonds = [c for c in cnei if len(cliques[c]) == 2]
        rings = [c for c in cnei if len(cliques[c]) > 2]
        if len(bonds) > 2 or (len(bonds) == 2 and len(cnei) > 2): #In general, if len(cnei) >= 3, a singleton should be added, but 1 bond + 2 ring is currently not dealt with.
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges[(c1,c2)] = 1
        elif len(rings) > 2: #Multiple (n>2) complex rings
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges[(c1,c2)] = 100 - 1
        else:
            for i in range(len(cnei)):
                for j in range(i + 1, len(cnei)):
                    c1,c2 = cnei[i],cnei[j]
                    inter = set(cliques[c1]) & set(cliques[c2])
                    if edges[(c1,c2)] < len(inter):
                        edges[(c1,c2)] = len(inter) #cnei[i] < cnei[j] by construction
    edges = [u + (100-v,) for u,v in edges.items()]
    return edges,  cliques

def get_motifs(mol):
    cliques, nei_list, n_atoms = get_cliques(mol)
    edges, cliques = get_cliques_edges(mol, cliques, nei_list, n_atoms)

    # Convert cliques to smiles
    cliques_smiles = []
    count = 1
    for clique in cliques:
        clique_mol = get_fragment_mol(mol, clique)
        cliques_smiles.append(get_smiles(clique_mol))
        count += 1

    return cliques_smiles, cliques, edges


