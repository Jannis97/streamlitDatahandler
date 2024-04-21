import csv
import sqlite3
from rdkit import Chem
from rdkit.Chem import Descriptors
import os

class MoleculeDatabase:
    def __init__(self, db_path, csv_path):
        self.db_path = db_path
        self.csv_path = csv_path
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.setup_database()

    def setup_database(self):
        """Erstellt die Datenbanktabelle, falls diese noch nicht existiert."""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS molecules (
                id INTEGER PRIMARY KEY,
                smiles TEXT,
                molecular_formula TEXT,
                molecular_weight REAL,
                logP REAL,
                atom_count INTEGER,
                bond_count INTEGER,
                aromatic_rings INTEGER
            )
        ''')
        self.conn.commit()

    def insert_molecule(self, data):
        """Fügt ein Molekül in die Datenbank ein."""
        self.cursor.execute('''
            INSERT INTO molecules (smiles, molecular_formula, molecular_weight, logP, atom_count, bond_count, aromatic_rings)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', data)
        self.conn.commit()

    def read_and_store(self):
        """Liest SMILES-Strings aus der CSV-Datei und speichert die Moleküldaten in der Datenbank."""
        with open(self.csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                smiles = row['smiles']
                molecule = Chem.MolFromSmiles(smiles)
                if molecule:
                    molecular_formula = Chem.rdMolDescriptors.CalcMolFormula(molecule)
                    molecular_weight = Descriptors.MolWt(molecule)
                    logP = Descriptors.MolLogP(molecule)
                    atom_count = molecule.GetNumAtoms()
                    bond_count = molecule.GetNumBonds()
                    aromatic_rings = len(Chem.GetSymmSSSR(molecule))
                    self.insert_molecule((smiles, molecular_formula, molecular_weight, logP, atom_count, bond_count, aromatic_rings))

    def close(self):
        """Schließt die Datenbankverbindung."""
        self.conn.close()

if __name__ == "__main__":
    current_dir = os.getcwd()
    csv_path = os.path.join(current_dir, 'data', 'smilesCSV', 'chemb0.csv')
    db_path = 'molecules.db'

    db = MoleculeDatabase(db_path, csv_path)
    db.read_and_store()
    db.close()

    print("Daten erfolgreich in die Datenbank eingefügt.")
