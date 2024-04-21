import streamlit as st
import pandas as pd
from sqlalchemy import create_engine


def main():
    st.title('Database Filtering Tool')

    uploaded_file = st.file_uploader("Upload your SQLite database", type=["sqlite", "db"])
    if uploaded_file is not None:
        # Save the uploaded database file to the current directory
        with open("temp_database.db", "wb") as f:
            f.write(uploaded_file.read())

        engine = create_engine('sqlite:///temp_database.db')
        st.success('Database uploaded successfully!')

        # Get a list of tables from the database
        query = "SELECT name FROM sqlite_master WHERE type='table';"
        tables = pd.read_sql(query, engine)

        table_names = tables["name"].tolist()
        selected_table = st.selectbox("Select a table to view", table_names)

        if selected_table:
            # Display the first few rows of the selected table
            data = pd.read_sql_table(selected_table, engine)
            st.write("Preview of selected table:")
            st.dataframe(data.head())

            # Allow the user to enter a SQL WHERE clause
            st.write("Enter a SQL WHERE clause to filter the data:")
            sql_filter = st.text_input("WHERE clause (e.g., column_name > 100)", "")

            if st.button('Apply Filter'):
                if sql_filter:
                    filtered_data = pd.read_sql(f"SELECT * FROM {selected_table} WHERE {sql_filter}", engine)
                    st.write("Filtered Data:")
                    st.dataframe(filtered_data)
                else:
                    st.error("Please enter a valid WHERE clause.")

    # Optional: Delete the temporary database file on closing the app or as needed
    # os.remove("temp_database.db")


if __name__ == "__main__":
    main()
