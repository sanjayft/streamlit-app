import streamlit as st
import pandas as pd
import pickle as pk

# Load model and scaler
model = pk.load(open('model.pkl', 'rb'))
scaler = pk.load(open('scaler.pkl', 'rb'))

st.header("Loan Prediction App")

# Inputs
no_of_dep = st.slider("Choose No of Dependents", 0, 5, key="dep")

grad = st.selectbox("Choose Education", 
                    ["Graduated", "Not Graduated"], 
                    key="education")

self_emp = st.selectbox("Self Employed?", 
                        ["Yes", "No"], 
                        key="self_emp")

annual_income = st.slider("Choose Annual Income", 
                          0, 10000000, 
                          key="annual_income")

loan_amount = st.slider("Choose Loan Amount", 
                        0, 10000000, 
                        key="loan_amount")

loan_dur = st.slider("Choose Loan Duration (Years)", 
                     0, 20, 
                     key="loan_dur")

cibil = st.slider("Choose CIBIL Score", 
                  0, 1000, 
                  key="cibil")

assets = st.slider("Choose Assets Value", 
                   0, 10000000, 
                   key="assets")

# Convert categorical to numeric
grad_s = 0 if grad == "Graduated" else 1
emp_s = 1 if self_emp == "Yes" else 0

# Prediction button
if st.button("Predict"):

    # Create DataFrame with REAL VALUES
    pred_data = pd.DataFrame([[no_of_dep, grad_s, emp_s,
                               annual_income, loan_amount,
                               loan_dur, cibil, assets]],
                             columns=['no_of_dependents',
                                      'education',
                                      'self_employed',
                                      'income_annum',
                                      'loan_amount',
                                      'loan_term',
                                      'cibil_score',
                                      'Assets'])

    # Scale data
    pred_data_scaled = scaler.transform(pred_data)

    # Predict
    prediction = model.predict(pred_data_scaled)

    # Show result
    if prediction[0] == 1:
        st.success("✅ Loan is Approved")
    else:
        st.error("❌ Loan is Rejected")