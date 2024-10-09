import os
import sys

import streamlit as st

# Add the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Display an image in the sidebar
st.sidebar.image(
    r"c:\Users\e.bahri\Desktop\Logo-galileo-color-2x.png",
    caption="Galileo Logo",
    use_column_width=True,
)


####################
def process_file_1(uploaded_file_crm_df, uploaded_file_answers_df):
    import os

    import pandas as pd

    from notebooks.modules.control_crm_va import (
        clean_columns_df,
        get_va_from_crm,
        prepare_answers,
        prepare_crm_df,
    )

    answers_df = prepare_answers(uploaded_file_answers_df)
    crm_ref = prepare_crm_df(uploaded_file_crm_df)

    answers_df, crm_ref = clean_columns_df(crm_ref, answers_df)

    crm_with_jury_date_4 = get_va_from_crm(crm_ref, answers_df)

    a = crm_with_jury_date_4.duplicated(
        subset=["nom_matching", "prenom_matching", "UC_matching"]
    ).sum()

    duplicates = crm_with_jury_date_4[
        crm_with_jury_date_4.duplicated(
            subset=["nom_matching", "prenom_matching", "UC_matching"], keep=False
        )
    ]

    duplicates_list = duplicates.to_dict(orient="records")
    duplicates_list = [dic["nom_matching"] for dic in duplicates_list]

    return crm_with_jury_date_4, a, duplicates_list


#####################
def process_file_2(uploaded_file_crm_with_jury_date_final_df, uploaded_file_answers_df):
    from notebooks.modules.control_crm_va import correct_va, merge_crm_va_and_answers

    output_matching = merge_crm_va_and_answers(
        uploaded_file_crm_with_jury_date_final_df, uploaded_file_answers_df
    )

    output_matching_2, result = correct_va(output_matching)

    return output_matching_2, result


#####################

st.title("Étape 4 : Contrôle CRM - Voies d’accès ")
st.markdown(
    """
Cette page est dédiée au contrôle et à la vérification des données extraites du CRM pour les voies d'accès, en comparaison avec les réponses post-traitement. Voici les étapes à suivre pour exécuter le Notebook 'control_crm_VA' :
"""
)

# Section 1: Prérequis
st.header("Prérequis")
st.markdown(
    """
Pour cette étape, assurez-vous de disposer des éléments suivants :
- L'extraction CRM du titre en question
- Le fichier 'answers postprocessed' au format CSV
"""
)

# Section 2: Exécution du Notebook
st.header("Instructions")
st.markdown(
    """
1. Renseignez l'extraction CRM et le fichier 'answers postprocessed'.
2. En sortie vous aurez :
   - le fichier 'answers corrigé'
   - le document à ajuster manuellement en cas de cas restants
   - le fichier contenant les différences entre le PV et le CRM

3.Lorsque l'execution s'arrête en cours de route avec le message "Tu dois faire à la main : x cas à traiter", ouvrez le document nommé 'crm_with_jury_date_4'.
4. Supprimez les lignes non pertinentes et renommez le fichier en 'crm_with_jury_date_final'.
5. Ce document contient toutes les personnes pour lesquelles les voies d'accès ont été extraites de la CRM. Si certaines personnes apparaissent deux fois, vous devez décider quelle ligne conserver et supprimer l'autre. Consultez Cynthia en cas de doute.
6. Le Notebook signale x cas de ce type où les personnes apparaissent deux fois.

7. Relancez la suite du code pour finaliser le processus.

Le fichier des différences entre le PV et le CRM inclura une colonne 'Différences'. Une croix ('X') indique une différence entre les informations du PV et celles de la CRM pour une personne donnée.
"""
)

st.markdown("---")


uploaded_file_crm = st.file_uploader("Ton extract crm format xslx", type=["xlsx"])

uploaded_file_answers = st.file_uploader("Ton answers", type=["csv"])


if uploaded_file_crm is not None and uploaded_file_answers is not None:
    import pandas as pd

    uploaded_file_crm_df = pd.read_excel(uploaded_file_crm, engine="openpyxl")

    uploaded_file_answers_df = pd.read_csv(uploaded_file_answers, sep=";")
    # Process the uploaded file
    crm_with_jury_date_4, a, duplicates_list = process_file_1(
        uploaded_file_crm_df, uploaded_file_answers_df
    )

    # Display the processed data
    st.write(f"Tu dois faire ça à la main : {a} cas à traiter ")
    st.write(duplicates_list)

    # Offer download button for processed data
    st.download_button(
        label="le dossier à corriger et à renvoyer",
        data=crm_with_jury_date_4.to_csv(sep=";", index=False),
        file_name="crm_with_jury_date_4.csv",
        mime="text/csv",
    )


###
st.markdown("---")
st.write("Seconde étape")

uploaded_file_crm_with_jury_date_final = st.file_uploader(
    "crm_with_jury_date_final", type=["csv"]
)


if uploaded_file_crm_with_jury_date_final is not None:
    uploaded_file_crm_with_jury_date_final_df = pd.read_csv(
        uploaded_file_crm_with_jury_date_final, sep=";"
    )

    output_matching_2, result = process_file_2(
        uploaded_file_crm_with_jury_date_final_df, uploaded_file_answers_df
    )

    st.download_button(
        label="answer corrigé",
        data=output_matching_2.to_csv(sep=";", index=False),
        file_name="corrected_answers.csv",
        mime="text/csv",
    )

    st.download_button(
        label="le fichier des différences",
        data=result.to_csv(sep=";", index=False),
        file_name="differences.csv",
        mime="text/csv",
    )


st.markdown("---")


st.header("Vérification des Voies d'accés")
st.markdown(
    """
Cette étape permet de s'assurer que aucune voie d'accèe est vide.
"""
)


uploaded_file_answers_2 = st.file_uploader("Ton answers corrigé", type=["csv"])


if uploaded_file_answers_2 is not None:
    import pandas as pd

    df = pd.read_csv(uploaded_file_answers_2, sep=";")

    df["Voie d'accès_final"] = df["Voie d'accès_manuel"].fillna(df["Voie d'accès"])
    df_problem = df[df["Voie d'accès_final"].isna()]

    if df_problem.empty:
        st.success("Tout est ok !")
    else:
        st.warning("Il y a des voies d'accès vides, les voici : ")
        st.write(df_problem)
