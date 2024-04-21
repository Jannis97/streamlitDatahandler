import streamlit as st
import sys
import pandas as pd

# Append the path of the classes directory to sys.path
sys.path.append('classes')

# Import the class after adding the directory to sys.path
from rdkitSmile2Molinfo import rdkitSmile2Molinfo


def main():
    st.title('SMILES Molecular Property Calculator')

    smiles_input = st.text_input("Enter a SMILES string:", "CCO")  # Example default SMILES for ethanol

    if st.button("Calculate Properties"):
        try:
            mol_info = rdkitSmile2Molinfo(smiles_input)
            properties = mol_info.get_all_properties()

            # Convert properties dictionary to a DataFrame for nicer display
            properties_df = pd.DataFrame(properties.items(), columns=['Property', 'Value'])
            st.write(properties_df)
        except ValueError as e:
            st.error(f"Error: {str(e)}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    main()
