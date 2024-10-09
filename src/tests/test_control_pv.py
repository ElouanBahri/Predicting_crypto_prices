import pandas as pd
import pytest

from src.notebooks.modules.control_pv import (
    concat_columns,  # Assurez-vous d'importer la fonction correctement
)


# Fixture pour créer un DataFrame de test
@pytest.fixture
def dataframe():
    data = {
        "col1": ["val1", "val2", "val3"],
        "col1_manuel": [None, "manual2", None],
        "col2": ["val4", "val5", "val6"],
        "col2_manuel": ["manual4", None, "manual6"],
    }
    df = pd.DataFrame(data)
    return df


def test_concat_columns(dataframe):
    df = dataframe
    list_cols = ["col1", "col2"]
    result_df = concat_columns(df, list_cols)

    expected_col1_final = ["val1", "manual2", "val3"]
    expected_col2_final = ["manual4", "val5", "manual6"]

    assert (
        result_df["col1_final"].tolist() == expected_col1_final
    ), f"Expected {expected_col1_final} but got {result_df['col1_final'].tolist()}"
    assert (
        result_df["col2_final"].tolist() == expected_col2_final
    ), f"Expected {expected_col2_final} but got {result_df['col2_final'].tolist()}"


##################

from src.notebooks.modules.control_pv import clean_columns_values


# Fixture pour créer un DataFrame de test
@pytest.fixture
def dataframe2():
    data = {
        "col1": ["ÀéioÙ", "Hellô\nWôrld", " PyThôn-Program "],
        "col2": ["Nïce", "Bâd-\nExample", "  Test--Case  "],
    }
    df = pd.DataFrame(data)
    return df


def test_clean_columns_values(dataframe2):
    df = dataframe2
    col_name_str_list = ["col1", "col2"]
    result_df = clean_columns_values(df, col_name_str_list)

    expected_col1 = ["aeiou", "hello world", "python program"]
    expected_col2 = ["nice", "bad example", "test case"]

    assert (
        result_df["col1"].tolist() == expected_col1
    ), f"Expected {expected_col1} but got {result_df['col1'].tolist()}"
    assert (
        result_df["col2"].tolist() == expected_col2
    ), f"Expected {expected_col2} but got {result_df['col2'].tolist()}"


#######

from src.notebooks.modules.control_pv import (
    change_datetime,  # Assurez-vous d'importer la fonction correctement
)


def test_change_datetime():
    # Cas 1: Date au format "YYYY-MM-DD HH:MM:SS"
    assert change_datetime("2023-11-28 14:30:00") == "28/11/2023"

    # Cas 2: Date au format "DD/MM/YYYY"
    assert change_datetime("28/11/2023") == "28/11/2023"

    # Cas 3: Date au format "DD/MM/YYYY HH:MM"
    assert change_datetime("28/11/2023 14:30") == "28/11/2023"

    # Cas

    assert change_datetime("03/05/2023") == "03/05/2023"

    # Cas 4: Date déjà au format "DD/MM/YYYY" sans heure
    assert change_datetime("01/01/2020") == "01/01/2020"

    # Cas 6: NaT string
    assert change_datetime("NaT") == "NaT"


###############################
from src.notebooks.modules.control_pv import give_errors


@pytest.fixture
def data():
    path_answers = r"C:\Users\e.bahri\Dropbox (GGEF)\Equipe Data\Nettoyage de données\Fichiers answers_postprocessed à importer\MIN\answers_postprocessed_b_20240701-1147_MIN_V5.csv"
    path_csv_pv = r"C:\Users\e.bahri\Dropbox (GGEF)\PV de Jury\Manager de l'innovation numérique\Web school Factory\PV 2023\PV JUry certification P2023 - 10 nov 2023 - signé(Elouan).csv"

    df_answers = pd.read_csv(path_answers, sep=";")
    df_pv = pd.read_csv(path_csv_pv, sep=";")

    df = give_errors(df_answers, df_pv, [2023])[1]

    return [df, df_answers, df_pv]


def test_give_errors(data):
    assert give_errors(data[1], data[2], [2023])[0].shape[0] == 132
    assert give_errors(data[1], data[2], [2023])[1].shape[0] == 68


##############################
from src.notebooks.modules.control_pv import automatic_filling


def test_automatic_filling(
    data,
):  # Vérifier que le answers de sortie et d'entrèe ont les memes dimensions
    df_answers = pd.read_csv(
        r"C:\Users\e.bahri\Dropbox (GGEF)\Equipe Data\Nettoyage de données\Fichiers answers_postprocessed à importer\MIN\answers_postprocessed_b_20240701-1147_MIN_V5.csv",
        sep=";",
    )

    result_df = automatic_filling(data[0], data[1], data[2])

    assert result_df.shape == df_answers.shape
