from rdkit import Chem
from rdkit.Chem import Descriptors, Draw


class MoleculeData:
    def __init__(self, smiles):
        self.smiles = smiles
        self.molecule = Chem.MolFromSmiles(smiles)
        self.calculate_properties()

    def calculate_properties(self):
        """Berechnet und speichert verschiedene chemische Eigenschaften des Moleküls."""
        self.molecular_formula = Chem.rdMolDescriptors.CalcMolFormula(self.molecule)
        self.molecular_weight = Descriptors.MolWt(self.molecule)
        self.atom_count = self.molecule.GetNumAtoms()
        self.bond_count = self.molecule.GetNumBonds()
        self.logP = Descriptors.MolLogP(self.molecule)
        self.aromatic_rings = len(Chem.GetSymmSSSR(self.molecule))  # Anzahl der aromatischen Ringe
        self.atoms = [{'atom_id': atom.GetIdx(), 'atom_type': atom.GetSymbol(),
                       'atomic_number': atom.GetAtomicNum(), 'charge': atom.GetFormalCharge()}
                      for atom in self.molecule.GetAtoms()]
        self.bonds = [{'bond_id': bond.GetIdx(), 'atom1_id': bond.GetBeginAtomIdx(),
                       'atom2_id': bond.GetEndAtomIdx(), 'bond_type': bond.GetBondType().name}
                      for bond in self.molecule.GetBonds()]

    def get_molecule_image(self):
        """Erzeugt ein Bild des Moleküls."""
        return Draw.MolToImage(self.molecule)

    def display_properties(self):
        """Gibt die gespeicherten Eigenschaften des Moleküls aus."""
        print(f"SMILES: {self.smiles}")
        print(f"Molecular Formula: {self.molecular_formula}")
        print(f"Molecular Weight: {self.molecular_weight}")
        print(f"Number of Atoms: {self.atom_count}")
        print(f"Number of Bonds: {self.bond_count}")
        print(f"LogP: {self.logP}")
        print(f"Aromatic Rings: {self.aromatic_rings}")
        print("Atoms:")
        for atom in self.atoms:
            print(atom)
        print("Bonds:")
        for bond in self.bonds:
            print(bond)


# Beispiel für die Verwendung der Klasse
if __name__ == "__main__":
    aspirin = "CC(=O)OC1=CC=CC=C1C(=O)O"
    molecule_data = MoleculeData(aspirin)
    molecule_data.display_properties()
    image = molecule_data.get_molecule_image()
    image.show()  # Zeigt das Bild des Moleküls an
