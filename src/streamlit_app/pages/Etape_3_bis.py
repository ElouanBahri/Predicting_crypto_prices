import os
import sys

import streamlit as st

# Display an image in the sidebar
st.sidebar.image(
    r"c:\Users\e.bahri\Desktop\Logo-galileo-color-2x.png",
    caption="Galileo Logo",
    use_column_width=True,
)


st.title(
    "Etape 3 bis:  Vérification des doublons et de la complétion des dates de Jury"
)

# Section 1: Vérification des doublons
st.header("Vérification des doublons indépendamment de la date de jury et de l'école")
st.markdown(
    """
Cette vérification est basée uniquement sur les noms, prénoms et certification des individus pour identifier les doublons potentiels dans les données.
"""
)

# Section 2: Vérification des dates de jury différentes
st.header("Vérification des dates de jury différentes pour la même personne")
st.markdown(
    """
Cette étape permet de s'assurer qu'une même personne ne présente pas des dates de jury différentes dans les données, ce qui pourrait indiquer une erreur ou une incohérence.
"""
)

# Section 3: Vérification des dates de jury renseignées
st.header("Vérification que toutes les dates de jury sont renseignées")
st.markdown(
    """
Il est crucial que toutes les dates de jury soient correctement renseignées pour assurer l'intégrité et la fiabilité des données analysées. Cette vérification garantit que aucune date de jury n'est manquante.
"""
)


# Add the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


#####################

st.markdown("---")

uploaded_file_answers = st.file_uploader("Ton answers", type=["csv"])


if uploaded_file_answers is not None:
    import pandas as pd

    from notebooks.modules.utils import (
        change_datetime,
        clean_columns_values,
        concat_columns,
    )

    raw_df_survey = pd.read_csv(uploaded_file_answers, delimiter=";")
    df_survey = raw_df_survey[raw_df_survey["To Remove"].isna()]

    cols_to_concat = ["Nom", "Prénom", "Ecole (UC)", "Promotion", "Date de Jury"]

    concat_columns(df_survey, cols_to_concat)

    st.write("1er contôle : Vérification que toutes les dates de jury sont renseignés")

    if df_survey["Date de Jury_final"].notna().all():
        st.success("The column 'Date de Jury_final' doesn't have any empty cells.")
    else:
        st.warning(
            "There is at least one empty cell in the column 'Date de Jury_final'."
        )

    df_survey["Date de Jury_final"] = df_survey.apply(
        lambda row: change_datetime(row["Date de Jury_final"]), axis=1
    )

    df_survey_cleaned = clean_columns_values(
        df_survey, ["Nom_final", "Prénom_final", "Titre/diplôme"]
    )

    duplicated_rows = df_survey[
        df_survey.duplicated(
            subset=["Nom_final", "Prénom_final", "Titre/diplôme", "Durée post-diplôme"],
            keep=False,
        )
    ]

    st.write(
        "2ème contôle : Vérification des doublons indépendamment de la date de jury et de l'école "
    )

    if duplicated_rows.empty:
        st.success("Tout est OK!")
    else:
        st.warning("Il y a des doublons, les voici :")
        st.write(duplicated_rows)

    df_problem_jury_date = df_survey_cleaned[
        df_survey_cleaned.duplicated(
            subset=["Nom_final", "Prénom_final", "Durée post-diplôme"], keep=False
        )
    ]
    df_problem_jury_date = df_problem_jury_date.groupby(
        ["Nom_final", "Prénom_final", "Durée post-diplôme"]
    ).filter(lambda x: x["Date de Jury_final"].nunique() > 1)
    df_problem_jury_date = df_problem_jury_date.sort_values(by="Nom_final")

    st.write(
        "3ème contôle :  Vérification des dates de jury différentes pour la même personne "
    )

    if df_problem_jury_date.empty:
        st.success("Tout est OK!")
    else:
        st.warning("Il y a des doublons, les voici :")
        st.write(df_problem_jury_date)
