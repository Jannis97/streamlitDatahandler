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

        # Erstelle ein Dictionary für jede Bindung
        bonds_dict = {}
        for bond in self.bonds:
            start_atom_idx = bond.GetBeginAtomIdx()
            end_atom_idx = bond.GetEndAtomIdx()
            bonds_dict[(start_atom_idx, end_atom_idx)] = {
                'start_atom_idx': start_atom_idx,
                'end_atom_idx': end_atom_idx,
                'start_position': self.positions[start_atom_idx],
                'end_position': self.positions[end_atom_idx],
                'color': 'gray',
                'width': 5
            }

        return atoms_dict, bonds_dict

    def analyze_molecule(self):
        smiles = self.smiles
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)  # Füge Wasserstoffe hinzu, um vollständige Informationen zu erhalten

        # Analyse der Hybridisierung der Atome
        atom_hybridization = {}
        for atom in mol.GetAtoms():
            atom_hybridization[atom.GetIdx()] = {
                'symbol': atom.GetSymbol(),
                'hybridization': atom.GetHybridization().name
            }

        # Analyse der Bindungstypen
        bond_types = {}
        for bond in mol.GetBonds():
            bond_types[(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())] = {
                'start_symbol': bond.GetBeginAtom().GetSymbol(),
                'end_symbol': bond.GetEndAtom().GetSymbol(),
                'bond_type': bond.GetBondType().name
            }

        return atom_hybridization, bond_types

    def create_3d_plot(self):
        atoms_dict, bonds_dict = self.generate_dicts()
        atom_hybridization, bond_types = self.analyze_molecule()
        # Erstelle Traces für Atome
        atom_traces = []
        for atom_id, atom_info in atoms_dict.items():
            atom_trace = go.Scatter3d(
                x=[atom_info['position'][0]],
                y=[atom_info['position'][1]],
                z=[atom_info['position'][2]],
                mode='markers+text',
                marker=dict(size=atom_info['size'], color=atom_info['color'], opacity=0.8),
                text=atom_info['symbol'],
                textposition='middle center',
                textfont=dict(size=16, color='black'),
                name=f'Atom {atom_id + 1}'
            )
            atom_traces.append(atom_trace)

        # Erstelle Traces für Bindungen
        bond_traces = []
        for (start_atom_idx, end_atom_idx), bond_info in bonds_dict.items():
            bond_trace = go.Scatter3d(
                x=[bond_info['start_position'][0], bond_info['end_position'][0], None],
                y=[bond_info['start_position'][1], bond_info['end_position'][1], None],
                z=[bond_info['start_position'][2], bond_info['end_position'][2], None],
                mode='lines',
                line=dict(color=bond_info['color'], width=bond_info['width']),
                name=f'Bond {start_atom_idx + 1}-{end_atom_idx + 1}'
            )
            bond_traces.append(bond_trace)

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
smiles_string = "CCO"  # Beispiel SMILES für Ethanol
visualizer = MoleculeVisualizer(smiles_string)
visualizer.create_3d_plot()
