import streamlit as st
import pandas as pd
from sqlalchemy import create_engine


def main():
    st.title('SMILES Length Enhancer')

    uploaded_file = st.file_uploader("Upload your SQLite database", type=["sqlite", "db"])
    if uploaded_file is not None:
        db_path = "temp_database.db"
        with open(db_path, "wb") as f:
            f.write(uploaded_file.read())
        engine = create_engine(f'sqlite:///{db_path}')
        st.success('Database uploaded successfully!')

        query = "SELECT name FROM sqlite_master WHERE type='table';"
        tables = pd.read_sql(query, engine)
        if tables.empty:
            st.error("No tables found in the database.")
            return

        selected_table = st.selectbox("Select a table", tables['name'])
        if selected_table:
            data = pd.read_sql_table(selected_table, engine)
            st.write("Preview of selected table:")
            st.dataframe(data.head())

            smiles_column = st.selectbox("Select the SMILES column", data.columns)
            if st.button("Calculate SMILES Length"):
                if smiles_column:
                    data['SMILES_Length'] = data[smiles_column].fillna('').apply(lambda x: len(str(x)))
                    st.dataframe(data[['SMILES_Length']].head())
                    data.to_sql(selected_table, con=engine, if_exists='replace', index=False)
                    st.success("Database updated with SMILES lengths.")
                    with open(db_path, "rb") as file:
                        st.download_button(
                            label="Download Modified Database",
                            data=file,
                            file_name='modified_database.db',
                            mime='application/x-sqlite3'
                        )


if __name__ == "__main__":
    main()
