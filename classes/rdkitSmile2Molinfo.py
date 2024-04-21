from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
from PIL import Image
import io

class rdkitSmile2Molinfo:
    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)
        if self.mol is None:
            raise ValueError("Invalid SMILES string.")

    def get_molecular_weight(self):
        return Descriptors.MolWt(self.mol)

    def get_num_rotatable_bonds(self):
        return Descriptors.NumRotatableBonds(self.mol)

    def get_logp(self):
        return Descriptors.MolLogP(self.mol)

    def get_num_h_donors(self):
        return Descriptors.NumHDonors(self.mol)

    def get_num_h_acceptors(self):
        return Descriptors.NumHAcceptors(self.mol)

    def count_aromatic_atoms(self):
        return sum(1 for atom in self.mol.GetAtoms() if atom.GetIsAromatic())

    def count_specific_atoms(self, symbol):
        return sum(1 for atom in self.mol.GetAtoms() if atom.GetSymbol() == symbol)

    def get_all_properties(self):
        return {
            "Molecular Weight": self.get_molecular_weight(),
            "Number of Rotatable Bonds": self.get_num_rotatable_bonds(),
            "LogP": self.get_logp(),
            "Number of Hydrogen Bond Donors": self.get_num_h_donors(),
            "Number of Hydrogen Bond Acceptors": self.get_num_h_acceptors(),
            "Number of Aromatic Atoms": self.count_aromatic_atoms(),
            "Number of Carbon Atoms": self.count_specific_atoms('C'),
            "Number of Hydrogen Atoms": self.count_specific_atoms('H'),
            "Number of Nitrogen Atoms": self.count_specific_atoms('N'),
            "Number of Sulfur Atoms": self.count_specific_atoms('S'),
            "Number of Fluorine Atoms": self.count_specific_atoms('F')
        }

    def draw_molecule(self):
        img = Draw.MolToImage(self.mol)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()
