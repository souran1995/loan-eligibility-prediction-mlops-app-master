import pandas as pd
import logging
from pathlib import Path
import pandas as pd
import os
import sys

import pickle

from sklearn.preprocessing import OneHotEncoder


def get_continuous_columns_in_dataframe(dataframe):
    continuous_columns = dataframe.select_dtypes(include='number').columns
    return continuous_columns

def get_categorical_columns(dataframe: pd.DataFrame):
    categorical_columns_list = list(dataframe.select_dtypes(include='object').columns)
    return categorical_columns_list

def preprocess(X: pd.DataFrame, models_dir: Path, training_mode: bool) -> pd.DataFrame:

    if 'Loan_ID' in X.columns:
        X.pop('Loan_ID')
    if 'Customer_ID' in X.columns:
        X.pop('Customer_ID')
    #X = X.drop(['Loan_ID', 'Customer_ID'], axis=1)
    #X = drop_cols_with_missing_vals(X, 50)
    X_preprocessed = preprocess_continuous_data(X)
    X_preprocessed = preprocess_categorical_data(X_preprocessed, models_dir, training_mode)
    return X_preprocessed


def drop_cols_with_missing_vals(X, percentage = 50):
    percent_missing = X.isnull().sum() * 100 / len(X)
    missing_value_df = pd.DataFrame({'column_name': X.columns, 'percent_missing': percent_missing})
    cols_to_drop = missing_value_df[missing_value_df['percent_missing'] > percentage]
    cols_to_drop = cols_to_drop['column_name'].tolist()
    return X.drop(cols_to_drop, axis=1)

## Continuous data ##
def preprocess_continuous_data(X):
    X_preprocessed = X.copy()
    continuous_columns = get_continuous_columns_in_dataframe(X_preprocessed)
    for column_name in continuous_columns:
        X_preprocessed[column_name] = X_preprocessed[column_name].fillna(X_preprocessed[column_name].mean())
    return X_preprocessed

## Categorical data ##
def preprocess_categorical_data(X: pd.DataFrame, models_dir: Path, training_mode: bool) -> pd.DataFrame:
    X_preprocessed = impute_missing_categorical_data(X)
    X_with_encoded_categorical_features = encode_categorical_features_orchestrator(X_preprocessed, models_dir, training_mode)
    return X_with_encoded_categorical_features

def impute_missing_categorical_data(X: pd.DataFrame) -> pd.DataFrame:
    X_preprocessed = X.copy()
    categorical_columns = get_categorical_columns(X_preprocessed)
    for column_name in categorical_columns:
        X_preprocessed[column_name] = X_preprocessed[column_name].fillna(X_preprocessed[column_name].mode()[0])
    return X_preprocessed

def encode_categorical_features_orchestrator(X: pd.DataFrame, models_dir: Path,
                                             training_mode: bool = False) -> pd.DataFrame:
    one_hot_encoder = get_categorical_encoder(X, models_dir, training_mode)
    X_with_continuous_data_and_encoded_categorical_data = encode_categorical_features(one_hot_encoder, X)
    return X_with_continuous_data_and_encoded_categorical_data

def get_categorical_encoder(X: pd.DataFrame, models_dir: Path, training_mode: bool):
    encoder_filepath = models_dir / 'encoder_categorical.pkl'
    #encoder_filepath = './models/encoder_categorical.pkl'
    if training_mode:
        logging.info('Generating a new encoder')
        categorical_columns = get_categorical_columns(X)
        one_hot_encoder = OneHotEncoder(handle_unknown='ignore', dtype=int, sparse=True)
        one_hot_encoder.fit(X[categorical_columns])
        with open(encoder_filepath, 'wb') as file:
            pickle.dump(one_hot_encoder, file)
    else:
        logging.info('Loading an existng encoder')
        if encoder_filepath.is_file():
           with open(encoder_filepath, 'rb') as file:  
               one_hot_encoder = pickle.load(file)
        else:
           raise Exception('Catergorical Encoder not found' + str(encoder_filepath))

    return one_hot_encoder

def encode_categorical_features(one_hot_encoder, X):
    categorical_columns = get_categorical_columns(X)
    continuous_columns = get_continuous_columns_in_dataframe(X)
    encoded_categorical_data_matrix = one_hot_encoder.transform(X[categorical_columns])
    encoded_data_columns = one_hot_encoder.get_feature_names(categorical_columns)
    encoded_categorical_data_df = pd.DataFrame.sparse.from_spmatrix(data=encoded_categorical_data_matrix,
                                                                    columns=encoded_data_columns, index=X.index)
    X_with_continuous_data_and_encoded_categorical_data = X.copy()[continuous_columns].join(encoded_categorical_data_df)
    return X_with_continuous_data_and_encoded_categorical_data



# if __name__ == '__main__':
#     ROOT_DIR = Path('./')
#     DATA_DIR = ROOT_DIR / 'data/house-prices'
#     MODELS_DIR = ROOT_DIR / 'models'
#     print("Working Directory: {}".format(os.getcwd()))
#     X = pd.read_excel('./data/loan_eligibility.xlsx')
#     X = preprocess(X, MODELS_DIR, training_mode=True)
#     print(X.head())