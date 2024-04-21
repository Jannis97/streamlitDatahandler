import streamlit as st
from rdkit import Chem
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def generate_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol

def generate_3d_graph(mol):
    try:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for bond in mol.GetBonds():
            start = bond.GetBeginAtom().GetIdx()
            end = bond.GetEndAtom().GetIdx()
            ax.plot([mol.GetConformer().GetAtomPosition(start)[0], mol.GetConformer().GetAtomPosition(end)[0]],
                    [mol.GetConformer().GetAtomPosition(start)[1], mol.GetConformer().GetAtomPosition(end)[1]],
                    [mol.GetConformer().GetAtomPosition(start)[2], mol.GetConformer().GetAtomPosition(end)[2]], color='black')

        ax.scatter([mol.GetConformer().GetAtomPosition(i)[0] for i in range(mol.GetNumAtoms())],
                   [mol.GetConformer().GetAtomPosition(i)[1] for i in range(mol.GetNumAtoms())],
                   [mol.GetConformer().GetAtomPosition(i)[2] for i in range(mol.GetNumAtoms())], color='b')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_title('3D Molecule Graph')

        st.pyplot(fig)
    except ValueError as e:
        st.error(f"Error generating 3D graph: {str(e)}")


def main():
    st.title("3D Molecule Graph Generator")

    # Eingabefeld für den SMILES-String
    smiles_input = st.text_input("Enter SMILES String", "")

    if smiles_input:
        # Generiere das Molekül aus dem SMILES-String
        mol = generate_molecule(smiles_input)

        if mol:
            # Generiere den 3D-Graphen des Moleküls
            generate_3d_graph(mol)
        else:
            st.error("Invalid SMILES string. Please enter a valid SMILES.")

if __name__ == "__main__":
    main()
