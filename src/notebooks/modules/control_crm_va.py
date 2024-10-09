import unicodedata

import numpy as np
import pandas as pd
from unidecode import unidecode


def change_datetime(date_jury_final):
    """Take a datetime string and formate it to '%d/%m/%Y'

    Parameters
    ----------
    date_jury_final : str
        the jury date of a student

    Returns
    -------
    str
        the formated date
    """
    if isinstance(date_jury_final, str) and date_jury_final != "NaT":
        if "/" not in date_jury_final:
            return pd.to_datetime(date_jury_final, format="%Y-%m-%d %H:%M:%S").strftime(
                "%d/%m/%Y"
            )
        elif len(date_jury_final) == 10:
            return pd.to_datetime(date_jury_final, format="%d/%m/%Y").strftime(
                "%d/%m/%Y"
            )
        else:
            return pd.to_datetime(date_jury_final, format="%d/%m/%Y %H:%M").strftime(
                "%d/%m/%Y"
            )
    return date_jury_final


def concat_columns(df: pd.DataFrame, list_cols: list):
    """concatenate the items in list_cols to create a new column, 'column_name'_final. It takes the manual value if not empty, else the original value.
    Parameters
    ----------
    df : DataFrame
        the answers_postprocessed dataframe
    list_cols : _type_
        the columns that we want to concatenate
    """
    for col_name in list_cols:
        df[col_name + "_final"] = df[col_name + "_manuel"].fillna(df[col_name])


def clean_column_values(data_df, col_name_str):
    """
    will clean column values from sepecial caracters
    :param data_df: forms output
    :param col_name_str: column name to clean
    :returns: dataframe with cleaned column
    """
    return (
        data_df[col_name_str]
        .astype(str)  # Convertir en chaîne de caractères
        .str.lower()  # Mettre en minuscules
        .apply(lambda x: unicodedata.normalize("NFKD", x))  # Normaliser
        .str.encode(
            "ascii", errors="ignore"
        )  # Encoder en ASCII en ignorant les erreurs
        .str.decode("utf-8")  # Décoder en UTF-8
        .str.replace("\n", "")  # Remplacer les sauts de ligne par par pas d'espace
        .str.strip()  # Supprimer les espaces au début et à la fin
        .str.replace(r"-", "", regex=True)  # Remplacer les tirets par par pas d'espace
        .str.replace(
            r"\s+", "", regex=True
        )  # Remplacer les espaces multiples par pas d'espace
        .apply(unidecode)  # Supprimer les accents restants
    )


def prepare_crm_df(crm_output):
    cols_crm_to_keep = [
        "UC",
        "Nom (Contact de l'opportunité)",
        "Prénom (Contact de l'opportunité)",
        "Courrier électronique (Contact de l'opportunité)",
        "Courrier électronique",
        "Saison",
        "Cursus",
        "Rythme / Financement",
        "Etat",
    ]

    crm_ref = crm_output[cols_crm_to_keep]

    crm_ref["Voie d'accès_crm"] = crm_ref.apply(
        lambda row: get_voie_acces(row["Rythme / Financement"], row["Etat"]), axis=1
    )
    return crm_ref


def get_voie_acces(rythme, etat):
    if rythme == "Classique":
        return "Etudiant (Formation initiale)"
    elif rythme == "Apprentissage":
        return "Apprenti (Contrat d’apprentissage)"
    elif rythme == "Alternance":
        return "En contrat de professionnalisation"
    elif rythme == "Formation continue":
        return "Stagiaire de la formation professionnelle (Formation continue)"
    elif etat in [
        "Alternant : Placé",
        "Alternant : Recherche aboutie",
        "Alternant : En suivi spécifique",
    ]:
        return "En contrat de professionnalisation"
    elif etat in ("Alternant : Sous réserve de placement", "Classique"):
        return "Etudiant (Formation initiale)"
    else:
        return "Autre"


promotion_list = [2019, 2020, 2021, 2022, 2023]


def prepare_answers(raw_answers_df: pd.DataFrame):
    raw_answers_df["Promotion_final"] = raw_answers_df["Promotion_manuel"].fillna(
        raw_answers_df["Promotion"]
    )

    raw_answers_df = raw_answers_df[
        raw_answers_df["Promotion_final"].isin(promotion_list)
    ]

    raw_answers_df = raw_answers_df.drop(columns="Promotion_final")

    cols_answers_to_keep = [
        "UNIQUE_KEY",
        "Nom",
        "Nom_manuel",
        "Prénom",
        "Prénom_manuel",
        "Ecole (UC)",
        "Ecole (UC)_manuel",
        "Promotion",
        "Promotion_manuel",
        "Email",
        "Email_manuel",
        "Téléphone",
        "Téléphone_manuel",
        "Voie d'accès",
        "Voie d'accès_manuel",
        "Date de Jury",
        "Date de Jury_manuel",
    ]
    answers_df = raw_answers_df.copy().filter(cols_answers_to_keep)

    cols_to_concat = [
        "Nom",
        "Prénom",
        "Ecole (UC)",
        "Promotion",
        "Voie d'accès",
        "Email",
        "Téléphone",
    ]
    concat_columns(answers_df, cols_to_concat)

    return answers_df


def clean_columns_df(crm_ref: pd.DataFrame, answers_df: pd.DataFrame):
    # Nettoyage des colonnes contenant les noms, prénoms et école
    crm_ref["nom_matching"] = clean_column_values(
        crm_ref, "Nom (Contact de l'opportunité)"
    )
    crm_ref["prenom_matching"] = clean_column_values(
        crm_ref, "Prénom (Contact de l'opportunité)"
    )
    answers_df["nom_matching"] = clean_column_values(answers_df, "Nom_final")
    answers_df["prenom_matching"] = clean_column_values(answers_df, "Prénom_final")
    crm_ref["UC_matching"] = clean_column_values(crm_ref, "UC")
    answers_df["UC_matching"] = clean_column_values(answers_df, "Ecole (UC)_final")

    for col in ["UC_matching", "nom_matching", "prenom_matching"]:
        crm_ref[col] = crm_ref[col].apply(lambda x: " ".join(x.split()))
        answers_df[col] = answers_df[col].apply(lambda x: " ".join(x.split()))

    return answers_df, crm_ref


def get_va_from_crm(crm_ref, answers_df):
    crm_ref["promo"] = crm_ref["Saison"].apply(lambda x: int(x.split("-")[1]))

    list_crm_groupby = [
        "UC_matching",
        "nom_matching",
        "prenom_matching",
    ]
    new_cols_filter = [
        "nom_matching",
        "prenom_matching",
        "UC",
        "UC_matching",
        "Nom (Contact de l'opportunité)",
        "Prénom (Contact de l'opportunité)",
        "Cursus",
        "Courrier électronique (Contact de l'opportunité)",
        "Courrier électronique",
        "Rythme / Financement",
        "Voie d'accès_crm",
        "promo",
    ]
    df_crm = pd.DataFrame(crm_ref.filter(new_cols_filter).drop_duplicates())

    df_answers_2 = answers_df.drop_duplicates(
        subset=["nom_matching", "prenom_matching", "UC_matching"]
    )

    df_answers_2["promotion_pv"] = df_answers_2["Promotion_manuel"].fillna(
        df_answers_2["Promotion"]
    )

    df_answers_2 = df_answers_2.drop(columns=["Promotion"])

    df_answers_2["Date de Jury_final"] = df_answers_2["Date de Jury_manuel"].fillna(
        df_answers_2["Date de Jury"]
    )

    df_answers_2["Date de Jury_final"] = df_answers_2.apply(
        lambda row: change_datetime(row["Date de Jury_final"]), axis=1
    )
    # On supprime les dates de jury vide
    df_answers_2 = df_answers_2[
        df_answers_2["Date de Jury_final"].apply(lambda x: isinstance(x, str))
    ]

    df_answers_2["Mois"] = df_answers_2.apply(
        lambda row: row["Date de Jury_final"].split("/")[1], axis=1
    )

    crm_with_jury_date = pd.merge(
        df_crm,
        df_answers_2,
        on=[
            "nom_matching",
            "prenom_matching",
            "UC_matching",
        ],
        how="inner",
    )

    # On garde que les promo qui sont égale à année de jury - 1 pour les date de Jury dans janvier et février
    condition_mois = crm_with_jury_date["Mois"].isin(["01", "02"])

    # Condition pour les promo égale à année de jury - 1, appliquée seulement si la première condition est vraie
    crm_with_jury_date["condition"] = np.where(
        condition_mois
        & (crm_with_jury_date["promo"] != crm_with_jury_date["promotion_pv"] - 1),
        False,
        True,
    )

    # Filtrer le DataFrame en utilisant la nouvelle colonne condition
    crm_with_jury_date_1 = crm_with_jury_date[crm_with_jury_date["condition"]]

    # Supprimer la colonne condition après filtrage
    crm_with_jury_date_1 = crm_with_jury_date_1.drop(columns=["condition"])

    crm_with_jury_date_2 = crm_with_jury_date_1[
        (crm_with_jury_date_1["promo"] == crm_with_jury_date_1["promotion_pv"])
        | (crm_with_jury_date_1["promo"] == crm_with_jury_date_1["promotion_pv"] - 1)
    ]

    crm_with_jury_date_3 = (
        crm_with_jury_date_2.groupby(list_crm_groupby)
        .apply(lambda x: x.drop_duplicates(subset="Voie d'accès_crm", keep="last"))
        .reset_index(drop=True)
    )

    crm_with_jury_date_3 = (
        crm_with_jury_date_3.groupby(list_crm_groupby)
        .apply(lambda x: x)
        .reset_index(drop=True)
    )

    crm_with_jury_date_4 = (
        crm_with_jury_date_3.groupby(list_crm_groupby)
        .apply(lambda x: x.drop_duplicates(subset="Cursus", keep="last"))
        .reset_index(drop=True)
    )

    return crm_with_jury_date_4


def merge_crm_va_and_answers(
    crm_with_jury_date_final: pd.DataFrame, raw_answers_df: pd.DataFrame
):
    crm_final = crm_with_jury_date_final[["UNIQUE_KEY", "Voie d'accès_crm"]]

    output_matching = pd.merge(raw_answers_df, crm_final, on=["UNIQUE_KEY"], how="left")

    return output_matching


def correct_va(output_matching):
    output_matching["Voie d'accès_PV"] = output_matching["Voie d'accès_manuel"].fillna(
        output_matching["Voie d'accès"]
    )

    output_matching["Difference"] = output_matching.apply(
        lambda row: "X" if row["Voie d'accès_PV"] != row["Voie d'accès_crm"] else "",
        axis=1,
    )

    result = output_matching[
        [
            "UNIQUE_KEY",
            "Difference",
            "Voie d'accès_PV",
            "Voie d'accès_crm",
            "Date de Jury",
            "Date de Jury_manuel",
            "Ecole (UC)",
            "Ecole (UC)_manuel",
        ]
    ]

    # if outpath000 != "":  # Pour le test
    #     result.to_csv(outpath000, index=False, sep=";")

    output_matching["Voie d'accès_manuel"] = output_matching["Voie d'accès_crm"].fillna(
        output_matching["Voie d'accès_PV"]
    )

    output_matching.drop(
        columns=[
            "Voie d'accès_crm",
            "Voie d'accès_PV",
            "Difference",
            "Promotion_final",
        ],
        axis=1,
        inplace=True,
    )

    return output_matching, result
