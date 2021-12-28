# loan-eligibility-prediction-mlops-app
This project is a machine learning MLOPS web app which predicts the loan eligibility based on a set of features.

Project WorkFlow:

1. Training Pipeline
2. Inference Pipeline
3. Developed Asynchronous API with FastAPI
4. Serve a ml model with FastAPI
5. UI using Streamlit for users to make predictions by calling FastAPI
6. Usage of Airflow to detect drift, retrain the model and make predictions by requesting FastAPI
7. Saving all the predictions in localdb (Postgres)
8. MLFlow to handle versioning of model objects after retraining and serving the correct model version in production

<h2>Project Setup</h2>

To setup this project on your system, first set the environment by running the pip install requirements.txt<br>
Once that is done without any errors, you can start FastAPI and Streamlit by running the following commands in the terminal:<br>
<h4>For FastAPI</h4>
uvicorn api.app:app --reload

<br>
It will start on port 8000 with localhost so probably on http://127.0.0.1:8000 or http://localhost:8000

<br>

<h4>For Streamlit</h4>
streamlit run streamlit/frontend.py

<br>
It will start on port 8501 on localhost so probably on http://127.0.0.1:8501 or http://localhost:8501

<br><br>

Now it's time to install and configure Airflow.

<h3>Airflow</h3>

<h4>For Windows</h4>
<b>1.</b> First, we need to download and install docker from the following link:<br>
https://docs.docker.com/get-docker/

<br><br>

<b>2.</b> Then, go to the project root and run the following command:<br>
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/2.2.3/docker-compose.yaml'<br>
This will download the docker compose file mentioning all the services needed by Airflow.

<br>
<b>3.</b> Now create the 2 new directories by running the following command:<br>
mkdir ./logs ./plugins

<br>

<b>4.</b> Now, you need to run the following the command:<br>
echo -e "AIRFLOW_UID=$(id -u)\nAIRFLOW_GID=0" > .env

<br>

<b>5.</b> Now, run the following command:<br>
docker-compose up airflow-init

<br>

<b>6.</b> Now, to finally run Airflow, run the following command:<br>
docker-compose up
<br>
Now, you can see the Airflow UI on port 8080 of your localhost so probably on http://127.0.0.1:8080 or http://localhost:8080

<br>
<h4>For Linux or Mac</h4>
Please follow the following link for Airflow installation:<br>
https://airflow.apache.org/docs/apache-airflow/stable/start/local.html

<br>
Please replace 'host.docker.internal' with 'localhost' in line 27,28 of prediction_dag and line 23,24 of drift_retraining_dag in the dags folder if not using docker.

<br>
<h4>DAGS Description</h4><br>
In our project, we have 3 DAGS:<br><br>
<b>1. </b> Ingestion DAG<br>
This dag will take a chunk of records (1000 rows in our case) from original dataset and save it in a new csv file with a timestamp.

<br><br>

<b>2. </b> Make Prediction Dag:<br>
This dag will be responsible for making the predictions on the ingested data and then saving the predictions in the postgres db by calling Fastapi.

<br>

<b>3. </b> Detect Drift and Re-train Model Dag:<br>
This dag will be responsible for detecting drift in the data and if there is, it will re-train the model. This newer model version and all the previous model versions will be managed by <b>MLFlow</b>. 

<br><br>
<b>(Note: Run these Dags in the same order as mentioned above.</b>)
<br>

<h4>MLFlow</h4>
MLFlow is used in the training part of the project and it is responsible for moving the right model version into the production as well in the models folder in the root.<br>
To start MLFlow, run the following command:<br><br>
mlflow server     --backend-store-uri sqlite:///mlruns.db     --default-artifact-root ../tmp/mlruns
<br><br>
Now, it will start MLFlow server on port 5000 of your localhost. So most probably, you will see be able to access the MLFlow UI from http://127.0.0.1:5000 or http://localhost:5000

<br><br>

<b>Important Notes:</b><br>
<b>1.</b> Please run MLFlow first before running the AIRFLOW Dags as the Retraining Dag depends upon MLFlow for model versioning.
<br>
<b>2.</b> Postgres PG Admin local credentials are in the .env file. Please change them according to your PG Admin local system.
