from pkg_resources import parse_version
import streamlit as st
import sys
import requests
from database import get_DAG_predictions, get_STREAMLIT_predictions 
import pandas as pd


sys.path.insert(0, '../api')
sys.path.insert(0, '../pipelines')

url = 'http://127.0.0.1:8000/predict'

#from pipelines import prediction, request_body
def main():
    st.title("Load Eligibility Check App")

    menu = ["Prediction","Past_Prediction_from_DAG", "Past_Prediction_from_STREAMLIT"]
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Prediction":
        st.subheader("Prediction APP")
        

        #result = print_arg(st.text_input('Overall Cond'))
        #st.write('Result: %s' % result)

        Loan_ID = st.text_input('Loan ID', 445412)
        Customer_ID = st.text_input('Customer ID', 123456)
        Current_Loan_Amount = st.text_input('Current Loan Amount', 445412)
        Term = st.selectbox('Loan Type',('Short Term', 'Long Term'))
        Credit_Score = st.text_input('Credit Score', 709)
        Annual_Income = st.text_input('Annual Income', 1167493)
        Years_in_current_job =st.text_input('Years in Current Job', "10 years")
        Home_Ownership = st.selectbox('Home Ownership',('Home Mortgage', 'Rent', 'Own Home'))
        Purpose = st.text_input('Purpose', "Debt Consolidation")
        Monthly_Debt = st.text_input('Monthly Debt', 5214.74)
        Years_of_Credit_History = st.text_input('Years of Credit History', 17.2)
        Months_since_last_delinquent = st.text_input('Months since last delinquent', 29)
        Number_of_Open_Accounts = st.text_input('Number of Open Accounts', 35)
        Number_of_Credit_Problems = st.text_input('Number of Credit Problems', 0)
        Current_Credit_Balance = st.text_input('Current Credit Balance', 228190)
        Maximum_Open_Credit = st.text_input('MAximum Open Credit', 416746)
        Bankruptcies = st.text_input('Bankruptcies', 1)
        Tax_Liens = st.text_input('Tax Liens', 0)

        mydata = {
            'Loan_ID' : Loan_ID,
            'Customer_ID' : Customer_ID,
            'Current_Loan_Amount' : Current_Loan_Amount,
            'Term' : Term,
            'Credit_Score' : Credit_Score,
            'Annual_Income' : Annual_Income,
            'Years_in_current_job' : Years_in_current_job,
            'Home_Ownership' : Home_Ownership,
            'Purpose' : Purpose,
            'Monthly_Debt' : Monthly_Debt,
            'Years_of_Credit_History' : Years_of_Credit_History,
            'Months_since_last_delinquent' : Months_since_last_delinquent,
            'Number_of_Open_Accounts' : Number_of_Open_Accounts,
            'Number_of_Credit_Problems' : Number_of_Credit_Problems,
            'Current_Credit_Balance' : Current_Credit_Balance,
            'Maximum_Open_Credit' : Maximum_Open_Credit,
            'Bankruptcies' : Bankruptcies,
            'Tax_Liens' : Tax_Liens,
            'prediction_type': "streamlit",
            'Loan_Status': "null"

        }


        if st.button('Make Prediction'):
            
            x = requests.post(url, json=mydata)
            st.success("Prediction for Loan Status: " + x.text)
            #model_prediction = prediction(overallCond, yrBuilt, yrSold)
            #st.write('Prediction: %s' % model_prediction)
            #st.write('Overall Cond: %s' % overallCond)
    elif choice == "Past_Prediction_from_DAG":
        st.subheader("Data Coming from PostgreSQL [Dag Predictions]")
        res = get_DAG_predictions()
        st.write(res)
        feat = ['Loan_ID' , 'Customer_ID' , 'Current_Loan_Amount' , 'Term', 'Credit_Score', 'Annual_Income', 'Years_in_current_job', 'Home_Ownership', 'Purpose', 'Monthly_Debt' , 'Years_of_Credit_History' , 'Months_since_last_delinquent' , 'Number_of_Open_Accounts', 'Number_of_Credit_Problems' ,'Current_Credit_Balance' ,'Maximum_Open_Credit' ,'Bankruptcies' ,'Tax_Liens' , 'Loan_status', 'Prediction_type' , 'Prediction_time']
        df = pd.DataFrame(res, columns=feat)
        st.dataframe(df)
    
    elif choice == "Past_Prediction_from_STREAMLIT":
        st.subheader("Data Coming from PostgreSQL")
        res = get_STREAMLIT_predictions()
        st.write(res)
        feat = ['Loan_ID' , 'Customer_ID' , 'Current_Loan_Amount' , 'Term', 'Credit_Score', 'Annual_Income', 'Years_in_current_job', 'Home_Ownership', 'Purpose', 'Monthly_Debt' , 'Years_of_Credit_History' , 'Months_since_last_delinquent' , 'Number_of_Open_Accounts', 'Number_of_Credit_Problems' ,'Current_Credit_Balance' ,'Maximum_Open_Credit' ,'Bankruptcies' ,'Tax_Liens' , 'Loan_status', 'Prediction_type' , 'Prediction_time']
        df = pd.DataFrame(res, columns=feat)
        st.dataframe(df)
        
    else:
        st.subheader("")
        

if __name__ == '__main__':
    main()
