import os
import sys

import pandas as pd
import streamlit as st

# Add the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Display an image in the sidebar
st.sidebar.image(
    r"c:\Users\e.bahri\Desktop\Logo-galileo-color-2x.png",
    caption="Galileo Logo",
    use_column_width=True,
)
#####################
from io import BytesIO


# Function to create an Excel workbook
def create_workbook(wb):
    # Save the workbook to a BytesIO object
    output = BytesIO()
    wb.save(output)
    output.seek(0)
    return output


#####################
def process_file(uploaded_file):
    import pandas as pd

    from notebooks.modules.pv_converter import (
        clean_table,
        create_df_final,
        extract_tables,
    )

    df, date = extract_tables(uploaded_file)
    result_0 = clean_table(df)
    result_1 = create_df_final(result_0, date)

    return result_1


#####################


st.title("TCD")


# Necessary files
st.markdown(
    """
Vous aurez besoin du : 

- PBI

"""
)

# Instructions
st.markdown(
    """
### Instructions pour le TCD :

1. Faites glisser votre PBI et remplissez les lignes qui apparaissent.
2. Faites de nouveau glisser votre PBI dans la deuxième case prévu à cet effet, et vous obtiendrez un fichier excel en sortie.
"""
)

st.markdown("---")

uploaded_file = st.file_uploader("Ton PBI", type=["csv"])


st.markdown("---")
if uploaded_file is not None:
    pbi_df = pd.read_csv(uploaded_file, sep=";")
    all_schools = pbi_df["Ecole (UC)"].unique().tolist()

    choice_1 = st.radio("6 mois ?", ("Oui", "Non"))

    # Determine the boolean value based on the user's choice
    result_1 = True if choice_1 == "Oui" else False
    if result_1:
        min_1 = st.number_input(
            "Salaire minimun :", min_value=0, max_value=10000000, value=0, step=1
        )
        max_1 = st.number_input(
            "Salaire maximun :", min_value=0, max_value=10000000, value=0, step=1
        )

        # List of options
        options = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]

        # Ask user to select multiple options
        selected_options_1 = st.multiselect(
            " Choisis les années que tu veux afficher", options
        )

        # Ask the user to select schools
        selected_schools_1 = st.multiselect(
            "Sélectionnez les écoles à prendre pour 6 mois (toutes incluses si vide):",
            all_schools,
        )
    else:
        min_1 = 5000
        max_1 = 150000
        selected_options_1 = []
        selected_schools_1 = []

    st.markdown("---")

    choice_2 = st.radio("Situation actuelle ?", ("Oui", "Non"))

    # Determine the boolean value based on the user's choice
    result_2 = True if choice_2 == "Oui" else False
    if result_2:
        min_2 = st.number_input(
            "Salaire minimun:", min_value=0, max_value=1000000, value=0, step=1
        )
        max_2 = st.number_input(
            "Salaire maximun:", min_value=0, max_value=1000000, value=0, step=1
        )

        # List of options
        options = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]

        # Ask user to select multiple options
        selected_options_2 = st.multiselect(
            "Choisis les années que tu veux afficher ici", options
        )

        # Ask the user to select schools
        selected_schools_2 = st.multiselect(
            "Sélectionnez les écoles à prendre pour situation actuelle (toutes incluses si vide):",
            all_schools,
        )
    else:
        min_2 = 5000
        max_2 = 150000
        selected_options_2 = []
        selected_schools_2 = []

    st.markdown("---")
    choice_3 = st.radio("Voie d'accès ?", ("Oui", "Non"))

    # Determine the boolean value based on the user's choice
    result_3 = True if choice_3 == "Oui" else False
    if result_3:
        # List of options
        options = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]

        # Ask user to select multiple options
        selected_options_3 = st.multiselect(
            "Choisis les années que tu veux corriger", options
        )

        # Ask the user to select schools
        selected_schools_3 = st.multiselect(
            "Sélectionnez les écoles (toutes incluses si vide):", all_schools
        )
    else:
        selected_options_3 = []
        selected_schools_3 = []

    st.markdown("---")
    choice_4 = st.radio("Source ?", ("Oui", "Non"))

    # Determine the boolean value based on the user's choice
    result_3 = True if choice_3 == "Oui" else False

st.markdown("---")

uploaded_file_1 = st.file_uploader("PBI", type=["csv"])

st.markdown("---")

if uploaded_file_1 is not None:
    from tqdm import tqdm

    from notebooks.modules.TCD import tcd

    processed_data = pd.read_csv(uploaded_file_1, sep=";", low_memory=False)

    result = tcd(
        processed_data,
        choice_1,
        min_1,
        max_1,
        selected_options_1,
        selected_schools_1,
        choice_2,
        min_2,
        max_2,
        selected_options_2,
        selected_schools_2,
        choice_3,
        selected_options_3,
        selected_schools_3,
        choice_4,
    )
    if st.button("Generate Workbook"):
        workbook = create_workbook(result)
        st.download_button(
            label="Download Excel Workbook",
            data=workbook,
            file_name="TCD.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
