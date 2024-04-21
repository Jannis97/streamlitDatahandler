import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import plotly.graph_objs as go

def generate_3d_graph(smiles):
    # Konvertiere den SMILES-String in ein RDKit-Molekül
    mol = Chem.MolFromSmiles(smiles)

    # Füge Wasserstoffatome hinzu
    mol = Chem.AddHs(mol)

    # Generiere die 3D-Konformation des Moleküls
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())

    # Extrahiere die Atompositionen aus der konformierten Molekülstruktur
    positions = mol.GetConformer().GetPositions()

    # Extrahiere die Bindungen zwischen den Atomen
    bonds = mol.GetBonds()

    # Extrahiere die Atomsymbole
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

    # Extrahiere die Anzahl der Valenzelektronen für jedes Atom
    valence_electrons = [atom.GetTotalValence() for atom in mol.GetAtoms()]

    # Extrahiere die x-, y- und z-Koordinaten der Atome
    x_coords = [positions[i][0] for i in range(len(positions))]
    y_coords = [positions[i][1] for i in range(len(positions))]
    z_coords = [positions[i][2] for i in range(len(positions))]

    # Erstelle eine Liste von Traces für die Atome
    atom_traces = []
    for i in range(len(x_coords)):
        color = 'black'
        if symbols[i] == 'C':
            color = 'black'
        elif symbols[i] == 'H':
            color = 'blue'
        elif symbols[i] == 'N':
            color = 'green'
        elif symbols[i] == 'O':
            color = 'red'
        size = valence_electrons[i] * 5  # Die Größe der Marker basiert auf der Anzahl der Valenzelektronen
        atom_trace = go.Scatter3d(
            x=[x_coords[i]],
            y=[y_coords[i]],
            z=[z_coords[i]],
            mode='markers+text',
            marker=dict(size=size, color=color, opacity=0.8),
            text=symbols[i],
            textposition='middle center',
            textfont=dict(size=16, color='black'),
            name=f'Atom {i + 1}'
        )
        atom_traces.append(atom_trace)

    # Erstelle eine Liste von Traces für die Bindungen
    bond_traces = []
    for bond in bonds:
        start_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()
        start_pos = positions[start_atom_idx]
        end_pos = positions[end_atom_idx]
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
    fig = go.Figure(data=atom_traces + bond_traces, layout=layout)

    # Zeige das interaktive 3D-Plot mit Plotly an
    st.plotly_chart(fig)


def main():
    st.title('Interactive 3D Molecule Visualization')

    # Eingabefeld für den SMILES-String
    smiles = st.text_input('Enter SMILES string')

    # Button zum Generieren der interaktiven 3D-Visualisierung
    if st.button('Generate Interactive 3D Visualization'):
        try:
            generate_3d_graph(smiles)
        except Exception as e:
            st.error(f'Error generating interactive 3D visualization: {str(e)}')


if __name__ == "__main__":
    main()
