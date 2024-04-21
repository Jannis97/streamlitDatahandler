#db_path = r'/chemblDb/chembl_34_sqlite/chembl_34/chembl_34_sqlite/chembl_34.db'
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine

def main():
    st.title('ChEMBL Database Viewer')

    # Assume you have a way to get the database path or URL
    database_path =  r'/home/jannis/repos/streamlitDatahandler/chemblDb/chembl_34_sqlite/chembl_34/chembl_34_sqlite/chembl_34.db'
    engine = create_engine(f'sqlite:///{database_path}')

    table_names = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", engine)['name'].tolist()
    selected_table = st.selectbox("Select a table to display", table_names)

    if selected_table:
        data = pd.read_sql_table(selected_table, engine)
        st.dataframe(data)

if __name__ == "__main__":
    main()
