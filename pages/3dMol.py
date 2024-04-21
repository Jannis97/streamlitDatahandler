import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
import numpy as np
import plotly.graph_objs as go

class MoleculeVisualizer:
    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)
        self.positions = None
        self.bonds = None
        self.symbols = None
        self.valence_electrons = None
        self.fig = None  # To store the plotly figure object

    def calculate_oxygen_bond_angles(self):
        # Calculate and return bond angles involving oxygen atoms
        angles = []
        for atom in self.mol.GetAtoms():
            if atom.GetSymbol() == 'O':
                idx = atom.GetIdx()
                # Get neighbors of the oxygen atom
                neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
                # Calculate angles if there are at least two neighbors
                if len(neighbors) > 1:
                    for i in range(len(neighbors)):
                        for j in range(i + 1, len(neighbors)):
                            angle = rdMolTransforms.GetAngleDeg(self.mol.GetConformer(), neighbors[i], idx, neighbors[j])
                            angles.append((idx, neighbors[i], neighbors[j], angle))
        return angles

    def generate_3d_graph(self):
        # Generiere die 3D-Konformation des Moleküls
        self.mol = Chem.AddHs(self.mol)
        AllChem.EmbedMolecule(self.mol, AllChem.ETKDG())

        # Extrahiere die Atompositionen aus der konformierten Molekülstruktur
        self.positions = self.mol.GetConformer().GetPositions()

        # Extrahiere die Bindungen zwischen den Atomen
        self.bonds = self.mol.GetBonds()

        # Extrahiere die Atomsymbole
        self.symbols = [atom.GetSymbol() for atom in self.mol.GetAtoms()]

        # Extrahiere die Anzahl der Valenzelektronen für jedes Atom
        self.valence_electrons = [atom.GetTotalValence() for atom in self.mol.GetAtoms()]

        # Erstelle eine Liste von Traces für die Atome
        atom_traces = []
        for i in range(len(self.positions)):
            color = 'black'
            if self.symbols[i] == 'C':
                color = 'black'
            elif self.symbols[i] == 'H':
                color = 'blue'
            elif self.symbols[i] == 'N':
                color = 'green'
            elif self.symbols[i] == 'O':
                color = 'red'
            size = self.valence_electrons[i] * 5  # Die Größe der Marker basiert auf der Anzahl der Valenzelektronen
            atom_trace = go.Scatter3d(
                x=[self.positions[i][0]],
                y=[self.positions[i][1]],
                z=[self.positions[i][2]],
                mode='markers+text',
                marker=dict(size=size, color=color, opacity=0.8),
                text=self.symbols[i],
                textposition='middle center',
                textfont=dict(size=16, color='black'),
                name=f'Atom {i + 1}'
            )
            atom_traces.append(atom_trace)

        # Erstelle eine Liste von Traces für die Bindungen
        bond_traces = []
        for bond in self.bonds:
            start_atom_idx = bond.GetBeginAtomIdx()
            end_atom_idx = bond.GetEndAtomIdx()
            start_pos = self.positions[start_atom_idx]
            end_pos = self.positions[end_atom_idx]
            x_start, y_start, z_start = start_pos
            x_end, y_end, z_end = end_pos
            bond_trace = go.Scatter3d(
                x=[x_start, x_end, None],
                y=[y_start, y_end, None],
                z=[z_start, z_end, None],
                mode='lines',
                line=dict(color='gray', width=5),
                name=f'Bond {start_atom_idx + 1}-{end_atom_idx + 1}'
            )
            bond_traces.append(bond_trace)

        # Erstelle das 3D-Figurenlayout
        layout = go.Layout(
            scene=dict(
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Z')
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            showlegend=True
        )

        # Erstelle das Plotly-Figurenobjekt
        self.fig = go.Figure(data=atom_traces + bond_traces, layout=layout)

        # Add bond angle annotations for oxygen
        oxygen_angles = self.calculate_oxygen_bond_angles()
        for angle in oxygen_angles:
            idx1, idx2, idx3, angle_value = angle
            mid_x = np.mean([self.positions[idx1][0], self.positions[idx2][0], self.positions[idx3][0]])
            mid_y = np.mean([self.positions[idx1][1], self.positions[idx2][1], self.positions[idx3][1]])
            mid_z = np.mean([self.positions[idx1][2], self.positions[idx2][2], self.positions[idx3][2]])
            self.fig.add_trace(go.Scatter3d(
                x=[mid_x],
                y=[mid_y],
                z=[mid_z],
                text=f'{angle_value:.2f}°',
                mode='text',
                showlegend=False
            ))

        # Zeige das interaktive 3D-Plot mit Plotly an
        st.plotly_chart(self.fig)

def main():
    st.title('Interactive 3D Molecule Visualization')

    # Eingabefeld für den SMILES-String
    smiles = st.text_input('Enter SMILES string')

    # Button zum Generieren der interaktiven 3D-Visualisierung
    if st.button('Generate Interactive 3D Visualization'):
        try:
            visualizer = MoleculeVisualizer(smiles)
            visualizer.generate_3d_graph()
        except Exception as e:
            st.error(f'Error generating interactive 3D visualization: {str(e)}')

if __name__ == "__main__":
    main()
