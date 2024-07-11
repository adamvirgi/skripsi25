import joblib
import streamlit as st

# Load the saved model and scaler
model = joblib.load('stunting_svm.pkl')
scaler = joblib.load('scaler.pkl')

# Define the application layout
st.title('Stunting Prediksi App')

# Get user input
gender = st.radio('gender', ['Male', 'Female'])
height = st.number_input('Tinggi Badan (cm)', min_value=0.0)
weight = st.number_input('Berat Badan (kg)', min_value=0.0)
gender = 1 if gender == 'Male' else 0 
# Preprocess the user input
input_data = [[gender, height, weight]]
input_scaled = scaler.transform(input_data)

# Make a prediction
if st.button('Predict'):
    prediction = model.predict(input_scaled)[0]
    index_labels = {
       1: 'Tinggi',
       2: 'Normal',
       3: 'Stunting',
       4: 'Severly Stunting',
       5: 'Extremely Stunting'
    }
    st.write(f'Predicted Stunting Category: {index_labels[prediction]}')

