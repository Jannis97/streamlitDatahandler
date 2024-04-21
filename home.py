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
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
