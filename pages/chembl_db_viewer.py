#db_path = r'/chemblDb/chembl_34_sqlite/chembl_34/chembl_34_sqlite/chembl_34.db'
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine


def main():
    st.title('ChEMBL Database Viewer')

    # Path to the ChEMBL SQLite database
    db_path = "pages/chembl_34.db"

    # Connect to the database using SQLAlchemy
    engine = create_engine(f'sqlite:///{db_path}')

    # Query to get all tables in the SQLite database
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    tables = pd.read_sql(query, engine)
    table_names = tables["name"].tolist()

    # Allow users to select a table to view
    selected_table = st.selectbox("Select a table to display", table_names)

    if selected_table:
        # Display the selected table
        query = f"SELECT * FROM {selected_table};"
        data = pd.read_sql(query, engine)
        st.write(f"Displaying table: {selected_table}")
        st.dataframe(data)


if __name__ == "__main__":
    main()
