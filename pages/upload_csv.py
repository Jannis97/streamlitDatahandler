import streamlit as st
import pandas as pd


def main():
    st.title('Upload and View CSV')

    # File uploader widget
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    # Select delimiter
    delimiter_option = st.selectbox(
        'Choose the delimiter that separates your data',
        options=[',', ';', '\t', '|', ' '],
        index=1  # Default to semicolon
    )

    # Select quote character
    quotechar_option = st.selectbox(
        'Choose the quotation character (used to wrap text fields)',
        options=['"', "'", '`'],
        index=0  # Default to double quote
    )

    if uploaded_file is not None:
        try:
            # Read CSV with selected delimiter and quote character
            data = pd.read_csv(
                uploaded_file,
                delimiter=delimiter_option,
                quotechar=quotechar_option
            )

            st.write("Select columns to display and download:")
            # Display checkboxes for each column in the DataFrame
            selected_columns = []
            for column in data.columns:
                # Create a checkbox for each column; if checked, add the column name to the list
                if st.checkbox(column, key=column):
                    selected_columns.append(column)

            if selected_columns:
                # Show a button to confirm the selection and display/download the selected data
                if st.button('Show Selected Columns'):
                    st.write("Data Preview:")
                    st.dataframe(data[selected_columns])

                    # Provide a download button for the selected data
                    if st.button('Download Selected Data', key='download'):
                        download_data(data, selected_columns)

        except Exception as e:
            st.error(f"An error occurred: {e}")


def download_data(data, selected_columns):
    """ Generate a download link allowing the selected data to be downloaded as a CSV file. """
    csv = data[selected_columns].to_csv(index=False)
    b64 = st.base64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="selected_data.csv">Download selected data as CSV</a>'
    st.markdown(href, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

