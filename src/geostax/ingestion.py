import pandas as pd


def read_data(paths):

    # Load Data

    df_collars = pd.read_excel(
        paths["collars"]["path"],
        sheet_name= paths["collars"]["sheet_name"],
    )
    df_survey = pd.read_excel(
        paths["survey"]["path"],
        sheet_name= paths["survey"]["sheet_name"],
    ).iloc[:, 0:4]
    df_assays = pd.read_excel(
        paths["assays"]["path"],
        sheet_name= paths["assays"]["sheet_name"],
    )

    return df_collars, df_survey, df_assays