import os
import sys

import pandas as pd
import pytest

# Add the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


from src.notebooks.modules.control_crm_va import (  # Assurez-vous d'importer la fonction correctement
    clean_columns_df,
    correct_va,
    get_va_from_crm,
    merge_crm_va_and_answers,
    prepare_answers,
    prepare_crm_df,
)


@pytest.fixture
def dataframe():
    answers_path = r"C:\Users\e.bahri\Dropbox (GGEF)\Equipe Data\Nettoyage de données\Fichiers answers_postprocessed à importer\CDG\answers_postprocessed\answers_postprocessed_b_20240610-1055_CDG_V17_Date_Jury_Done.csv"  # Changer avec le chemin du fichier answers_postprocessed
    crm_output_path_1 = (
        r"C:\Users\e.bahri\Desktop\CRM_CDG.xlsx"  # chemin du fichier CRM
    )
    crm_with_jury_date_final = pd.read_csv(
        r"C:\Users\e.bahri\Desktop\Test_control_crm\crm_with_jury_date_final.csv",
        sep=";",
    )

    raw_answers_df = pd.read_csv(answers_path, sep=";")
    answers_df = prepare_answers(raw_answers_df)
    crm_output_1 = pd.read_excel(crm_output_path_1, engine="openpyxl")
    crm_ref = prepare_crm_df(crm_output_1)
    answers_df, crm_ref = clean_columns_df(crm_ref, answers_df)
    output_matching = merge_crm_va_and_answers(crm_with_jury_date_final, raw_answers_df)

    return (
        crm_ref,
        answers_df,
        crm_with_jury_date_final,
        raw_answers_df,
        output_matching,
    )


######################################
@pytest.fixture
def dataframe2():
    answers_path = r"C:\Users\e.bahri\Dropbox (GGEF)\Equipe Data\Nettoyage de données\Fichiers answers_postprocessed à importer\MIN\Resultat_script\answers_postprocessed_b_20240703-0933_MIN_V6_PVc.csv"  # Changer avec le chemin du fichier answers_postprocessed
    crm_output_path_1 = r"C:\Users\e.bahri\Dropbox (GGEF)\Equipe Data\Nettoyage de données\Fichiers answers_postprocessed à importer\MIN\extract_crm_min.xlsx"  # chemin du fichier CRM
    crm_with_jury_date_final = pd.read_csv(
        r"C:\Users\e.bahri\Desktop\Test_control_crm\crm_with_jury_date_final_2.csv",
        sep=";",
    )

    raw_answers_df = pd.read_csv(answers_path, sep=";")
    answers_df = prepare_answers(raw_answers_df)
    crm_output_1 = pd.read_excel(crm_output_path_1, engine="openpyxl")
    crm_ref = prepare_crm_df(crm_output_1)
    answers_df, crm_ref = clean_columns_df(crm_ref, answers_df)
    output_matching = merge_crm_va_and_answers(crm_with_jury_date_final, raw_answers_df)

    return (
        crm_ref,
        answers_df,
        crm_with_jury_date_final,
        raw_answers_df,
        output_matching,
    )


######################################
def test_get_va_from_crm(dataframe):
    crm_with_jury_date_4 = get_va_from_crm(dataframe[0], dataframe[1])

    assert crm_with_jury_date_4.shape[0] == 850 + 9


def test_merge_crm_va_and_answers(dataframe):
    assert (
        merge_crm_va_and_answers(dataframe[2], dataframe[3]).shape[0]
        == dataframe[3].shape[0]
    )

    assert (
        merge_crm_va_and_answers(dataframe[2], dataframe[3]).shape[1]
        == dataframe[3].shape[1] + 1
    )


def test_correct_va(dataframe):
    assert correct_va(dataframe[4])[0].shape == (3006, 174)


######################################


def test_get_va_from_crm_2(dataframe2):
    crm_with_jury_date_4 = get_va_from_crm(dataframe2[0], dataframe2[1])

    assert crm_with_jury_date_4.shape[0] == 251 + 0


def test_merge_crm_va_and_answers_2(dataframe2):
    assert (
        merge_crm_va_and_answers(dataframe2[2], dataframe2[3]).shape[0]
        == dataframe2[3].shape[0]
    )

    assert (
        merge_crm_va_and_answers(dataframe2[2], dataframe2[3]).shape[1]
        == dataframe2[3].shape[1] + 1
    )


def test_correct_va_2(dataframe2):
    assert correct_va(dataframe2[4])[0].shape == (714, 174)
