from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
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

    def get_all_properties(self):
        return {
            "Molecular Weight": self.get_molecular_weight(),
            "Number of Rotatable Bonds": self.get_num_rotatable_bonds(),
            "LogP": self.get_logp(),
            "Number of Hydrogen Bond Donors": self.get_num_h_donors(),
            "Number of Hydrogen Bond Acceptors": self.get_num_h_acceptors()
        }

    def draw_molecule(self):
        img = Draw.MolToImage(self.mol)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()

