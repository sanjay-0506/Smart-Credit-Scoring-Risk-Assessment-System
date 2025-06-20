import streamlit as st
import pandas as pd
import joblib
import numpy as np


model = joblib.load('credit_risk_model.pkl')
scaler = joblib.load('scaler_model.pkl')
pca = joblib.load('pca_model.pkl')


st.title("Credit Default Risk Prediction App")
st.write("Upload customer financial data to predict default probability using a trained ML model.")


uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    
    data = pd.read_csv(uploaded_file)

    cols_to_drop = ['SeriousDlqin2yrs', 'Unnamed: 0']
    data = data.drop(columns=[col for col in cols_to_drop if col in data.columns])


    data['MonthlyIncome'].fillna(data['MonthlyIncome'].median(), inplace=True)
    data['NumberOfDependents'].fillna(0, inplace=True)

    data['BalanceToIncome'] = data['DebtRatio'] * data['MonthlyIncome']
    data['CreditLinePerDependent'] = data['NumberOfOpenCreditLinesAndLoans'] / (data['NumberOfDependents'] + 1)

    st.subheader("🔍 Uploaded Data Preview")
    st.dataframe(data.head())

    
    data_scaled = scaler.transform(data)

   
    pca_components = pca.transform(data_scaled)
    pca_df = pd.DataFrame(pca_components, columns=[f'pc{i+1}' for i in range(5)], index=data.index)

    
    combined_df = pd.DataFrame(data_scaled, columns=data.columns, index=data.index)
    final_input = pd.concat([combined_df, pca_df], axis=1)

    

    
    pred_proba = model.predict_proba(final_input)[:, 1]
    predictions = (pred_proba > 0.46).astype(int)

    
    result_df = data.copy()
    result_df['Default Probability'] = pred_proba
    result_df['Prediction (1 = Default)'] = predictions

    st.subheader("Prediction Results")
    st.dataframe(result_df[['Default Probability', 'Prediction (1 = Default)']])

    
    st.download_button(
        label="Download Results as CSV",
        data=result_df.to_csv(index=False).encode('utf-8'),
        file_name="credit_risk_predictions.csv",
        mime='text/csv'
    )
