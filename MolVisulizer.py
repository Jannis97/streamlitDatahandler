from rdkit import Chem
from rdkit.Chem import AllChem
import plotly.graph_objs as go
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import plotly.graph_objs as go
class MoleculeVisualizer:
    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)
        self.add_hydrogens_and_generate_3d_coordinates()

    def add_hydrogens_and_generate_3d_coordinates(self):
        # Füge Wasserstoffatome zum Molekül hinzu
        self.mol = Chem.AddHs(self.mol)
        # Generiere 3D-Koordinaten für das Molekül
        AllChem.EmbedMolecule(self.mol, AllChem.ETKDG())
        self.positions = self.mol.GetConformer().GetPositions()
        self.bonds = self.mol.GetBonds()
        self.symbols = [atom.GetSymbol() for atom in self.mol.GetAtoms()]

    def generate_dicts(self):
        # Erstelle ein Dictionary für jeden Knoten (Atom)
        atoms_dict = {}
        for i, pos in enumerate(self.positions):
            atoms_dict[i] = {
                'symbol': self.symbols[i],
                'position': pos,
                'index': i,
                'color': {'C': 'black', 'H': 'blue', 'N': 'green', 'O': 'red'}.get(self.symbols[i], 'gray'),
                'size': 30
            }

        bonds_dict = {}
        for i, bond in enumerate(self.bonds):
            start_atom_idx = bond.GetBeginAtomIdx()
            end_atom_idx = bond.GetEndAtomIdx()
            start_position = np.array(self.positions[start_atom_idx])
            end_position = np.array(self.positions[end_atom_idx])
            vector = end_position - start_position
            length = np.linalg.norm(vector)
            normalized_vector = vector / length if length != 0 else np.array([0,0,0])

            bonds_dict[(start_atom_idx, end_atom_idx)] = {
                'index': i,
                'start_atom_idx': start_atom_idx,
                'end_atom_idx': end_atom_idx,
                'start_position': start_position.tolist(),
                'end_position': end_position.tolist(),
                'color': 'gray',
                'width': 5,
                'length': length,
                'normalized_vector': normalized_vector.tolist()
            }

        return atoms_dict, bonds_dict

    def analyze_molecule(self):
        # Analyse der Hybridisierung der Atome und Bindungstypen
        atom_hybridization = {}
        bond_types = {}
        for atom in self.mol.GetAtoms():
            atom_hybridization[atom.GetIdx()] = {
                'symbol': atom.GetSymbol(),
                'hybridization': atom.GetHybridization().name
            }

        for bond in self.mol.GetBonds():
            bond_types[(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())] = {
                'start_symbol': bond.GetBeginAtom().GetSymbol(),
                'end_symbol': bond.GetEndAtom().GetSymbol(),
                'bond_type': bond.GetBondType().name
            }

        return atom_hybridization, bond_types

    def combine_dicts(self, atoms_dict, atom_hybridization, bonds_dict, bond_types):
        # Aktualisiere die Dictionaries mit zusätzlichen Analyseinformationen
        for idx, atom in atom_hybridization.items():
            if idx in atoms_dict:
                atoms_dict[idx].update(atom)

        for idx, bond in bond_types.items():
            if idx in bonds_dict:
                bonds_dict[idx].update(bond)

        return atoms_dict, bonds_dict

    def calculate_bond_angles(self):
        # Erstelle ein Dictionary für die Atome und Bindungen
        atoms_dict, bonds_dict = self.generate_dicts()

        angles_mat = np.zeros((len(bonds_dict), len(bonds_dict)))
        # Gehe durch jedes Atom und berechne die Winkel zwischen den Bindungen
        for atom_idx in atoms_dict:
            connected_bonds = [bond_key for bond_key in bonds_dict if atom_idx in bond_key]

            # Wenn weniger als zwei Bindungen vorhanden sind, kann kein Winkel berechnet werden
            if len(connected_bonds) < 2:
                continue

            # Berechne die Winkel zwischen allen Bindungen, die am Atom ansetzen
            for i in range(len(connected_bonds)):
                for j in range(i + 1, len(connected_bonds)):
                    bond1_key = connected_bonds[i]
                    bond2_key = connected_bonds[j]
                    start0 = bond1_key[0]
                    end0 = bond1_key[1]
                    start1 = bond2_key[0]
                    end1 = bond2_key[1]
                    connectedAtoms = set([start0, end0, start1, end1])

                    if len(connectedAtoms) < 4:
                        bond0 = bonds_dict[bond1_key]
                        bond1 = bonds_dict[bond2_key]
                        vec1 = np.array(bond0['normalized_vector'])
                        vec2 = np.array(bond1['normalized_vector'])

                        # Berechne den Winkel zwischen den Vektoren
                        dot_product = np.dot(vec1, vec2)
                        angle = np.arccos(np.clip(dot_product, -1.0, 1.0)) * 180 / np.pi

                        idx0 = bond0['index']
                        idx1 = bond1['index']
                        angles_mat[idx0, idx1] = angle
                        angles_mat[idx1, idx0] = angle
        return angles_mat

    def find_adjacent_bond(self):
        atoms_dict = self.atoms_dict
        bonds_dict = self.bonds_dict
        angle_matrix = self.calculate_bond_angles()
        for key in atoms_dict.keys():
            index = atoms_dict[key]['index']
            bondsCounter = 0
            adjBonds = {}
            for (begin, end) in bonds_dict.keys():
                if begin == index or end == index:
                    print(index, begin, end)
                    adjBonds[bondsCounter] = (begin, end)

                    bondsCounter += 1
            atoms_dict[key].update(adjBonds)

        for key, atom in atoms_dict.items():
            # Zähle die Anzahl der Bindungen für jedes Atom
            bond_count = sum(1 for (begin, end) in bonds_dict if begin == key or end == key)
            hybridization = atom['hybridization']

            symbol = atom['symbol']
            if hybridization == 'SP3' and bond_count == 2:
                # ether
                print(f"Atom {symbol}{key} mit SP3 Hybridisierung hat genau 2 Bindungen")
                a = atom[0]
                b = atom[1]

                idx0 = bonds_dict[a]['index']
                idx1 = bonds_dict[b]['index']
                angle = angle_matrix[idx0, idx1]
                print(f"Angle between bonds: {angle:.2f} degrees")


    def create_3d_plot(self):
        atoms_dict, bonds_dict = self.generate_dicts()
        atom_hybridization, bond_types = self.analyze_molecule()

        atoms_dict, bonds_dict = self.combine_dicts(atoms_dict, atom_hybridization, bonds_dict, bond_types)
        self.atoms_dict = atoms_dict
        self.bonds_dict = bonds_dict

        self.find_adjacent_bond()

        # Erstelle Traces für Atome und Bindungen
        atom_traces = [go.Scatter3d(
            x=[info['position'][0]],
            y=[info['position'][1]],
            z=[info['position'][2]],
            mode='markers+text',
            marker=dict(size=info['size'], color=info['color'], opacity=0.8),
            text=f"{info['symbol']} ({info['hybridization']})",
            textposition='middle center',
            textfont=dict(size=16, color='black'),
            name=f"Atom {info['index'] + 1}"
        ) for info in atoms_dict.values()]

        bond_traces = [go.Scatter3d(
            x=[info['start_position'][0], info['end_position'][0], None],
            y=[info['start_position'][1], info['end_position'][1], None],
            z=[info['start_position'][2], info['end_position'][2], None],
            mode='lines',
            line=dict(color=info['color'], width=info['width']),
            name=f"{info['start_symbol']}-{info['end_symbol']} ({info['bond_type']})"
        ) for info in bonds_dict.values()]

        # Erstelle das Plotly-Figurenobjekt
        layout = go.Layout(
            scene=dict(
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Z')
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            showlegend=True
        )
        fig = go.Figure(data=atom_traces + bond_traces, layout=layout)
        fig.show()

# Beispiel zur Verwendung der Klasse
smiles_string = "CO"  # Beispiel SMILES für Ethanol
visualizer = MoleculeVisualizer(smiles_string)
visualizer.create_3d_plot()
