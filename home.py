# source .venv/bin/activate
# streamlit run home.py

# echo fs.inotify.max_user_watches=524288 | sudo tee -a /etc/sysctl.conf
# sudo sysctl -p

# home.py
# home.py
import streamlit as st
import pandas as pd


def main():
    st.title('CSV File Uploader and Viewer')

    # Instructions
    st.write(
        "Upload a CSV file using the below widget. The file should be delimited by semicolons (`;`) and use double quotes (`\"`) as quotation marks.")

    # File uploader widget
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Assuming the file is encoded with the proper format
        try:
            # Read CSV file with specific delimiter and quoting character
            data = pd.read_csv(uploaded_file, delimiter=';', quotechar='"')
            st.write("Data Preview:")
            # Displaying the DataFrame in the app
            st.dataframe(data)
        except Exception as e:
            # Print the exception error message in the Streamlit interface
            st.error(f"An error occurred: {e}")
            st.write(
                "Please check that your file is properly formatted with semicolons as delimiters and double quotes for text.")

        # Option to download the displayed DataFrame
        if st.button('Download CSV'):
            # Convert DataFrame to CSV and generate a download button
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name='downloaded_data.csv',
                mime='text/csv'
            )


if __name__ == "__main__":
    main()
