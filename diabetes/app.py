import joblib
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the trained model and the StandardScaler object
model = joblib.load('diabetes_DT_model.joblib')
scaler = joblib.load('scaler.joblib')


# Create a Streamlit app
st.set_page_config(layout="wide")
col1, col2 = st.columns([2, 1])

col2.subheader("Details")


with col1:
    st.title('Diabetes Predictor')
    st.markdown('''<h4 style='text-align: left; color: #d73b5c;'> Select the features in the left panel and then click in the button below </h4>''', unsafe_allow_html=True)

    st.sidebar.markdown("## Features")



    # define the input fields
    age = st.sidebar.number_input("Age", min_value=16, max_value=90, value= 16, step=1)
    gender = st.sidebar.selectbox("Gender", ("-", "Male", "Female"), index=0)
    polyuria = st.sidebar.selectbox("Polyuria", ("-", "Yes", "No"), index=0)
    polydipsia = st.sidebar.selectbox("Polydipsia", ("-", "Yes", "No"), index=0)
    sudden_weight_loss = st.sidebar.selectbox("Sudden weight loss", ("-", "Yes", "No"), index=0)
    weakness = st.sidebar.selectbox("Weakness", ("-", "Yes", "No"), index=0)
    polyphagia = st.sidebar.selectbox("Polyphagia", ("-", "Yes", "No"), index=0)
    genital_thrush = st.sidebar.selectbox("Genital thrush", ("-", "Yes", "No"), index=0)
    visual_blurring = st.sidebar.selectbox("Visual blurring", ("-", "Yes", "No"), index=0)
    itching = st.sidebar.selectbox("Itching", ("-", "Yes", "No"), index=0)
    irritability = st.sidebar.selectbox("Irritability", ("-", "Yes", "No"), index=0)
    delayed_healing = st.sidebar.selectbox("Delayed healing", ("-", "Yes", "No"), index=0)
    partial_paresis = st.sidebar.selectbox("Partial paresis", ("-", "Yes", "No"), index=0)
    muscle_stiffness = st.sidebar.selectbox("Muscle stiffness", ("-", "Yes", "No"), index=0)
    alopecia = st.sidebar.selectbox("Alopecia", ("-", "Yes", "No"), index=0)
    obesity = st.sidebar.selectbox("Obesity", ("-", "Yes", "No"), index=0)


    if st.button("Make Prediction"):
        if '-' in [gender, polyuria, polydipsia, sudden_weight_loss, weakness, polyphagia, 
                genital_thrush, visual_blurring, itching, irritability, delayed_healing, 
                partial_paresis, muscle_stiffness, alopecia, obesity]:
            st.warning("Please select an option for each of the features.")
        else:
            data = {
                "age": [age],
                "gender": [1 if gender == "Male" else 0],
                "polyuria": [1 if polyuria == "Yes" else 0],
                "polydipsia": [1 if polydipsia == "Yes" else 0],
                "sudden_weight_loss": [1 if sudden_weight_loss == "Yes" else 0],
                "weakness": [1 if weakness == "Yes" else 0],
                "polyphagia": [1 if polyphagia == "Yes" else 0],
                "genital_thrush": [1 if genital_thrush == "Yes" else 0],
                "visual_blurring": [1 if visual_blurring == "Yes" else 0],
                "itching": [1 if itching == "Yes" else 0],
                "irritability": [1 if irritability == "Yes" else 0],
                "delayed_healing": [1 if delayed_healing == "Yes" else 0],
                "partial_paresis": [1 if partial_paresis == "Yes" else 0],
                "muscle_stiffness": [1 if muscle_stiffness == "Yes" else 0],
                "alopecia": [1 if alopecia == "Yes" else 0],
                "obesity": [1 if obesity == "Yes" else 0]
            }

            df = pd.DataFrame.from_dict(data)
            scaled_age = scaler.transform(df.loc[:, 'age'].values.reshape(-1, 1))
            df.iloc[:, 0] = scaled_age.flatten()
            #print(df)

            # Print the contents of df
            #st.write(df)
            

            # Make predictions
            prediction = model.predict(df)
            prediction_prob = model.predict_proba(df)
            
            # Dispay the predictions
            if prediction[0] == 1:
                st.write("Positive - The probability for the patient to have diabetes is", round(prediction_prob[0, 1] * 100, 1), "%")
            else:
                st.write("Negative - The probability for the patient to not have diabetes is", round(prediction_prob[0, 0] * 100, 1), "%")


with col2:
    st.info(
        "##### Machine learning model: *Random Forest Classifier*\n"
        "##### Best Model parameters:\n - max_depth=10")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", "100%")
    col2.metric("Precision", "100%")
    col3.metric("Recall", "100%")
    col4.metric("F1-Score", "100%")

    
    ft_cols = ['age',
               'gender',
               'polyuria',
               'polydipsia',
               'sudden_weight_loss',
               'weakness',
               'polyphagia',
               'genital_thrush',
               'visual_blurring',
               'itching',
               'irritability',
               'delayed_healing',
               'partial_paresis',
               'muscle_stiffness',
               'alopecia',
               'obesity']
    
    importance = model.feature_importances_
    df_importance = pd.DataFrame({'features': ft_cols, 'importance': importance})

    fig =plt.figure(figsize=(10,6))
    sns.barplot(x='importance', y='features', data=df_importance.sort_values(by='importance', ascending=False), orient='h')
    plt.title('Feature Importance', fontsize=16)
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Features', fontsize=14)
    plt.tick_params(labelsize=12)
    st.pyplot(fig)
    