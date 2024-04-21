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
            if st.button('Select Columns'):
                # Allow user to select columns to display
                selected_columns = st.multiselect(
                    'Select columns to display',
                    options=data.columns.tolist(),
                    default=data.columns.tolist()  # Default to all columns
                )

                # Display only selected columns
                st.write("Data Preview:")
                st.dataframe(data[selected_columns])
            else:
                # Display full dataframe
                st.write("Data Preview:")
                st.dataframe(data)

        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
