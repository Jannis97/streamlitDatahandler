import streamlit as st
import pandas as pd


def main():
    st.title('Upload and View CSV')

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    delimiter_option = st.selectbox('Choose the delimiter that separates your data', options=[',', ';', '\t', '|', ' '],
                                    index=1)
    quotechar_option = st.selectbox('Choose the quotation character', options=['"', "'", '`'], index=0)

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, delimiter=delimiter_option, quotechar=quotechar_option)
        st.write("Select columns to display and download:")

        selected_columns = [col for col in data.columns if st.checkbox(col, key=col)]
        if selected_columns:
            displayed_data = data[selected_columns]
            st.dataframe(displayed_data)

            # Convert DataFrame to CSV and then to string
            csv = displayed_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Selected Data",
                data=csv,
                file_name='selected_data.csv',
                mime='text/csv',
                key='download-selected'
            )


if __name__ == "__main__":
    main()
