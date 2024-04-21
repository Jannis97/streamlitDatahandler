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

    # Slider zur Auswahl des Limits
    limit = st.slider("Select the number of rows to display", 1, 1000, 501)

    # SQL-Abfrage zur Auswahl der ersten 'limit' Zeilen der ausgewählten Tabelle
    query = f'''
    SELECT *
    FROM {selected_table}
    LIMIT {limit}
    '''

    # Ausführen der SQL-Abfrage und Abrufen der Ergebnisse als DataFrame
    results_df = pd.read_sql(query, engine)

    # Anzeigen des DataFrames in der Streamlit-App
    st.dataframe(results_df)

    # Drucken der Spaltennamen
    st.write("Column Names:")
    st.write(results_df.columns.tolist())

    # Anzeigen der Anzahl der Zeilen
    st.write(f"Number of Rows: {results_df.shape[0]}")

if __name__ == "__main__":
    main()
