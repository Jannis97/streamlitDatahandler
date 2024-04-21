import streamlit as st
import pandas as pd
from sqlalchemy import create_engine

def main():
    st.title('Compound Structures Query')

    # Option zum Hochladen einer CSV-Datei mit SMILES
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file is not None:
        # DataFrame aus der hochgeladenen CSV-Datei erstellen
        smiles_df = pd.read_csv(uploaded_file)

        # Ersten Eintrag im DataFrame überspringen
        smiles_df = smiles_df.iloc[1:]

        # Anzeigen des DataFrames mit den hochgeladenen SMILES
        st.write("Uploaded SMILES:")
        st.dataframe(smiles_df)

        # Eingabe der Anzahl der SMILES
        num_smiles = st.number_input("Enter the number of SMILES to query", min_value=1, value=1)

        if st.button("Run Query"):
            # Verbindung zur SQLite-Datenbank
            engine = create_engine('sqlite:///pages/chembl_34.db')

            # Liste für Ergebnisse
            results = []

            for smile in smiles_df["canonical_smiles"].head(num_smiles):
                # SQL-Abfrage zur Auswahl der entsprechenden Zeile aus der Tabelle "compound_structures"
                query = f'''
                SELECT *
                FROM compound_structures
                WHERE canonical_smiles = ?
                '''

                # Ausführen der SQL-Abfrage und Abrufen der Ergebnisse als DataFrame
                result_df = pd.read_sql_query(query, engine, params=(smile,))
                results.append(result_df)

            # Ergebnisse zusammenfügen
            final_result_df = pd.concat(results)

            # Anzeigen des finalen DataFrames in der Streamlit-App
            st.dataframe(final_result_df)

            # Anzeigen der Anzahl der abgerufenen Zeilen
            st.write(f"Number of Rows: {final_result_df.shape[0]}")

            # Link zum Herunterladen der CSV-Datei
            st.markdown(get_csv_download_link(final_result_df), unsafe_allow_html=True)

            # Link zum Herunterladen der neuen Datenbank
            st.markdown(get_db_download_link(final_result_df), unsafe_allow_html=True)

def get_csv_download_link(df):
    csv = df.to_csv(index=False)
    href = f'<a href="data:file/csv;base64,{csv}" download="compound_structures.csv">Download CSV file</a>'
    return href

import sqlite3
import base64
from io import BytesIO

def get_db_download_link(df):
    # Verbindung zu einer temporären SQLite-Datenbank im Speicher herstellen
    conn = sqlite3.connect(":memory:")
    # DataFrame in die temporäre Datenbank schreiben
    df.to_sql("compound_structures", conn, index=False, if_exists="replace")
    # Datenbankinhalt in eine BytesIO-Datei schreiben
    buffer = BytesIO()
    for line in conn.iterdump():
        buffer.write(f"{line}\n".encode("utf-8"))
    buffer.seek(0)
    # Link zum Herunterladen der Datenbankdatei zurückgeben
    href = f'<a href="data:application/octet-stream;base64,{base64.b64encode(buffer.getvalue()).decode()}" download="new_database.sql">Download New Database</a>'
    return href




if __name__ == "__main__":
    main()
