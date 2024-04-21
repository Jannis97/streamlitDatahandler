import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import sqlite3
import os

def main():
    st.title('CSV to SQLite Converter')

    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(data.head())

        db_name = "converted_db.sqlite"
        if st.button('Convert CSV to SQLite'):
            create_sqlite_db(data, db_name)
            st.success('Database created successfully!')

            # Provide download link for the SQLite database
            with open(db_name, "rb") as file:
                btn = st.download_button(
                    label="Download SQLite Database",
                    data=file,
                    file_name=db_name,
                    mime="application/x-sqlite3"
                )

def create_sqlite_db(data, db_name):
    if os.path.exists(db_name):
        os.remove(db_name)
    engine = create_engine(f'sqlite:///{db_name}')
    data.to_sql('data', con=engine, index=False, if_exists='replace')

if __name__ == "__main__":
    main()
