import pandas as pd
import numpy as np
import pickle
#import joblib
import shutil
from pathlib import Path
from datetime import datetime
import os

import mlflow
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from pipelines.preprocess import preprocess

def train_model(training_data_filepath: str, models_dir: Path, retrain: int = 0) -> dict:
    """
    training_data_filepath (str): the path to the training data
    :returns (dict) {"model_performance": "", "model_path": ""} 
    """
    mlflow.set_tracking_uri('http://127.0.0.1:5000')

    with mlflow.start_run(run_name=f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"):
        
        mlflow.sklearn.autolog()

        if retrain != 0:
            #read the ingested data with drift
            filename = os.path.basename(training_data_filepath)
            X = pd.read_csv('./dags/data/output_data/'+filename)    
        else:
            X = pd.read_excel(training_data_filepath)

        y = X.pop('Loan_Status')
        y_preprocessed = y.replace({'Fully Paid': 1, 'Charged Off': 0})

        X_preprocessed = preprocess(X, models_dir, True)

        mlflow.log_params({"nb_samples": len(X_preprocessed)})
        x_test, y_test = train_job(X_preprocessed, y_preprocessed)
        run = mlflow.active_run().info
        # score = evaluate_model(x_test, y_test, run.artifact_uri)
        # mlflow.log_metrics({"accuracy": score})

    model_version = register_model_to_registry("KNN", run.run_id)
    transition_model_to_a_new_stage(model_version, "production", models_dir)

    # return {"model_performance": score, "model_path": run.artifact_uri}
    return {"model_path": run.artifact_uri}

def train_job(X: pd.DataFrame, y: pd.Series) -> tuple:
    x_train, x_test, y_train, y_test = train_test_split(X, y)
    x_train, x_test = scale_features(x_train, x_test)
    # model_path = fit_save_model(models_dir, x_train, y_train)
    _ = fit_save_model(x_train, y_train)
    # return x_test, y_test, model_path
    return x_test, y_test

def scale_features(train: np.array, test: np.array) -> tuple:
    scaler = StandardScaler()
    scaler.fit(train)
    x_train_scaled = scaler.transform(train)
    x_test_scaled = scaler.transform(test)
    return x_train_scaled, x_test_scaled

def fit_save_model(x: list, y: list) -> KNeighborsClassifier:
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(x, y)

    # model_filepath = models_dir / 'model.joblib'
    # with open(model_filepath, 'wb') as file:
    #    pickle.dump(classifier, file)
    return classifier

def evaluate_model(x_test: np.array, y_test: np.array, model_uri: Path) -> float:
    model_path = model_uri + '/model.pkl'
    print('evaluate model path: {}'.format(model_path))
    if Path(model_path).is_file():
        with open(model_path, 'rb') as file:  
            model = pickle.load(file)
    else:
        raise Exception('model not found')

    ypred = model.predict(x_test)
    score = compute_accuracy(y_test, ypred)
    return score

def compute_accuracy(y_test: np.ndarray, y_pred: np.ndarray, precision: int = 2) -> float:
    acc = accuracy_score(y_test, y_pred)
    return round(acc, precision)

def register_model_to_registry(model_name_in_model_registry: str, run_id: int):
    model_version = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=model_name_in_model_registry)
    return model_version

def transition_model_to_a_new_stage(model_version, stage: str, models_dir: Path, archive_existing_versions: bool = True):
    client = MlflowClient()
    updated_model_version = client.transition_model_version_stage(
        name=model_version.name, version=model_version.version,
        stage=stage, archive_existing_versions=archive_existing_versions
    )
    source = model_version.source + '/model.pkl'
    destination = models_dir
    new_path = shutil.copy2(source, destination)
    print('Model moved to production: {}'.format(new_path))
    return updated_model_version


if __name__ == '__main__':
    import os
    print(os.getcwd())
    ROOT_DIR = Path('../')
    MODELS_DIR = ROOT_DIR / 'models'
    response = train_model('../data/loan_eligibility.xlsx', MODELS_DIR)
    print(response)