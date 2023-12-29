import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from IPython.display import Image
from rdkit.Chem.Draw import rdMolDraw2D

def smile_to_tree_structure_image(smile):
    mol = Chem.MolFromSmiles(smile)
    AllChem.Compute2DCoords(mol)

    img = Draw.MolToImage(mol, kekulize=False)

    img.save('/home/bioinfor3/Lxh/multiCom/result/other/molPic/tree.png')

def atom_features(atom):
    return [atom.GetAtomicNum(), atom.GetTotalNumHs(), atom.GetImplicitValence(), atom.GetIsAromatic()]

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    AllChem.Compute2DCoords(mol)

    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature)

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    img = Draw.MolToImage(mol)
    img.save('/home/bioinfor3/Lxh/multiCom/result/other/molPic/mol.png')
    img.show()

    
    return c_size, features, edge_index

smiles_example = "CC(C)COC(=O)C(C)C"
c_size, features, edge_index = smile_to_graph(smiles_example)
smile_to_tree_structure_image(smiles_example)