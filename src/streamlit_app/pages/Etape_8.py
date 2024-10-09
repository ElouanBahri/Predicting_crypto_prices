import os
import sys
from typing import Tuple, Union

import numpy as np
import pandas as pd
import streamlit as st

# Display an image in the sidebar
st.sidebar.image(
    r"c:\Users\e.bahri\Desktop\Logo-galileo-color-2x.png",
    caption="Galileo Logo",
    use_column_width=True,
)


def process_file(uploaded_file_answers_df):
    import pandas as pd

    from notebooks.modules.verif import check_fv, check_salary

    fv = check_fv(uploaded_file_answers_df)
    salary = check_salary(uploaded_file_answers_df)

    return fv, salary


st.title("Etape 8 :  Diverses  vérifications")


# Add the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


#####################

st.markdown("---")

uploaded_file_answers = st.file_uploader("Ton answers", type=["csv"])


if uploaded_file_answers is not None:
    import pandas as pd

    uploaded_file_answers_df = pd.read_csv(uploaded_file_answers, sep=";")

    fv, salary = process_file(uploaded_file_answers_df)

    st.write("1er contôle : vérifs des fonctions visées ")

    if fv[0]:
        st.success("Tout est ok !")
    else:
        st.warning("Il y a un problème sur les fontions visées :")
        st.write(fv[1][["Nom", "Prénom", "Cible", "Cible_manuel"]])
    st.write("2ème contôle : vérifs des salaires")

    if salary[0]:
        st.success("Tout est ok !")
    else:
        st.warning("Il y a un problème sur les salaires :")
        st.write(salary[1])
