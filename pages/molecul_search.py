import streamlit as st
import pandas as pd
from sqlalchemy import create_engine

# Datenbankverbindung herstellen
db_path = "pages/chembl_34.db"
engine = create_engine(f'sqlite:///{db_path}')

import streamlit as st
import pandas as pd
from sqlalchemy import create_engine

def main():
    st.title('ChemBL Table Viewer')

    # Liste der relevanten Tabellennamen
    relevant_tables = [
        "assays",
        "compound_records",
        "compound_properties",
        "compound_structures",
        "activities",
        "molecule_dictionary",
        "target_dictionary"
    ]

    # Dropdown-Menü zur Auswahl der Tabelle
    selected_table = st.selectbox("Select a table", relevant_tables)

    # Verbindung zur SQLite-Datenbank
    engine = create_engine('sqlite:///pages/chembl_34.db')

    # SQL-Abfrage zur Auswahl der ersten 501 Zeilen der ausgewählten Tabelle
    query = f'''
    SELECT *
    FROM {selected_table}
    LIMIT 501
    '''

    # Ausführen der SQL-Abfrage und Abrufen der Ergebnisse als DataFrame
    results_df = pd.read_sql(query, engine)

    # Anzeigen des DataFrames in der Streamlit-App
    st.dataframe(results_df)

if __name__ == "__main__":
    main()
