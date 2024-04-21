import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

def main():
    st.title('Chemical Structure Viewer')

    smiles_input = st.text_input("Enter a SMILES string:")
    if st.button("Show Structure"):
        mol = Chem.MolFromSmiles(smiles_input)
        if mol:
            img = Draw.MolToImage(mol)
            st.image(img, use_column_width=True)
        else:
            st.error("Invalid SMILES string provided.")

if __name__ == "__main__":
    main()
