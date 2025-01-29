# Libraries
import pandas as pd
import numpy as np
import streamlit as st
import sklearn
import pickle

# streamlit app
st.title("Student Performence Prediction")

# input field
Hours_Studied=st.number_input("Hours Studied", min_value=0, max_value=20, value=0)
Attendance=st.number_input("Attendance %", min_value=0, max_value=100, value=0)
Access_to_Resources_m=st.selectbox("Access to Resources",['Low','Medium','High'])
Motivation_Level_m=st.selectbox("Motivation Level",['Low','Medium','High'])

# input data as dict
input_data={
    'Hours_Studied':Hours_Studied,
    'Attendance':Attendance,
    'Access_to_Resources_m':Access_to_Resources_m,
    'Motivation_Level_m':Motivation_Level_m
}

# Convert into DataFrame
new_data=pd.DataFrame([input_data])

# Mapping
LMH={
    'Low':1,
    'Medium':2,
    'High':3}

new_data['Access_to_Resources_m']=new_data['Access_to_Resources_m'].map(LMH)
new_data['Motivation_Level_m']=new_data['Motivation_Level_m'].map(LMH)

# Prediction model (LinearRegression)
with open('model.pkl','rb')as model_f:
    model=pickle.load(model_f)
    
# Prediction
Prediction=model.predict(new_data)

# Output
if st.button('Predict'):
    st.write(Prediction)