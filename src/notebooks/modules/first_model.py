import pandas as pd 


def convert_X_data(X : pd.DataFrame) -> pd.DataFrame :

    X["trade"] = X["trade"].map({False: 0, True: 1})
    X["action"] = X["action"].map({"A": 0, "D": 1, "U" : 2})
    X["side"] = X["side"].map({"A": 1, "B": 0})
    X = X.sort_values(by="obs_id").reset_index(drop=True)

    return X

def change_y_data(y : pd.DataFrame) -> pd.DataFrame : 
    
    n = 100 

    data_duplicated = pd.concat([y] * n, ignore_index=True)

    data_duplicated = data_duplicated.sort_values(by="obs_id").reset_index(drop=True)

    data_duplicated = data_duplicated.drop(columns=["obs_id"])

    return data_duplicated
