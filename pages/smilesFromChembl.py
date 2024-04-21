import streamlit as st
import pandas as pd
from sqlalchemy import create_engine

def download_smiles_csv():
    # Verbindung zur SQLite-Datenbank
    engine = create_engine('sqlite:///pages/chembl_34.db')

    # SQL-Abfrage zur Auswahl aller SMILES aus der Datenbank
    query = '''
    SELECT canonical_smiles
    FROM compound_structures
    '''

    # Ausführen der SQL-Abfrage und Abrufen der Ergebnisse als DataFrame
    smiles_df = pd.read_sql(query, engine)

    # CSV-Datei erstellen
    smiles_csv = smiles_df.to_csv(index=False)

    # Herunterladen der CSV-Datei
    st.download_button(
        label="Download SMILES as CSV",
        data=smiles_csv,
        file_name='chembl_smiles.csv',
        mime='text/csv'
    )

def main():
    st.title('Download SMILES as CSV')

    st.write("Click the button below to download all SMILES from the database as a CSV file.")
    download_smiles_csv()

if __name__ == "__main__":
    main()
