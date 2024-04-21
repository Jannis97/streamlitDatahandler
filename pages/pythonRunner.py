import streamlit as st
import traceback
import sys
from io import StringIO

def main():
    st.title("Python Code Runner")

    # Textarea für den Python-Code
    code = st.text_area("Enter Python code here", height=200)

    # Button zum Ausführen des Codes
    if st.button("Run Code"):
        # Python-Code im try-except-Block ausführen, um Fehler abzufangen
        try:
            # Output umleiten, um Konsolenausgabe zu erfassen
            old_stdout = sys.stdout
            redirected_output = sys.stdout = StringIO()

            # Code ausführen
            exec(code)

            # Konsolenausgabe anzeigen
            st.code(redirected_output.getvalue(), language="python")

            # Output zurücksetzen
            sys.stdout = old_stdout
        except Exception as e:
            # Fehlermeldung anzeigen
            st.error(f"An error occurred: {e}")
            # Fehler-Stacktrace anzeigen
            st.code(traceback.format_exc(), language="python")

        # Variablen anzeigen
        variables = get_variables(code)
        for var_name, var_value in variables.items():
            st.write(f"{var_name}: {var_value}")

def get_variables(code):
    # Führe den Code aus, um die Variablen zu erhalten
    variables = {}
    exec(code, variables)
    return {name: value for name, value in variables.items() if not name.startswith("__")}

if __name__ == "__main__":
    main()
