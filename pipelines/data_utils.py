import pandas as pd

def get_continuous_columns_in_dataframe(dataframe):
    continuous_columns = dataframe.select_dtypes(include='number').columns
    return continuous_columns

def get_categorical_columns(dataframe: pd.DataFrame):
    categorical_columns_list = list(dataframe.select_dtypes(include='object').columns)
    return categorical_columns_list