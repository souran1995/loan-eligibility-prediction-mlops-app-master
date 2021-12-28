import psycopg2
import base64
from io import BytesIO
import json
conn = psycopg2.connect(user='postgres',
                                password='1234',
                                host='localhost',
                                database='prediction_result')

def get_DAG_predictions():
  # Create a cursor
  cur = conn.cursor()
  # Define the query
  sql = """select * from future_value WHERE prediction_type = 'dag';"""
  # Perform the query
  cur.execute(sql)
  # Get the predictions
  predictions = cur.fetchall()
  # Commit and close
  conn.commit()     
  cur.close()
  # Return the predictions
  return predictions


def get_STREAMLIT_predictions():
  # Create a cursor
  cur = conn.cursor()
  # Define the query
  sql = """select * from future_value WHERE prediction_type = 'streamlit';"""
  # Perform the query
  cur.execute(sql)
  # Get the predictions
  predictions = cur.fetchall()
  # Commit and close
  conn.commit()     
  cur.close()
  # Return the predictions
  return predictions

