import joblib
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the trained model and the StandardScaler object
model = joblib.load('diabetes_DT_model.joblib')
scaler = joblib.load('scaler.joblib')


# Create a Streamlit app
st.title('Diabetes Predictor')

# define the input fields
age = st.slider("Age", min_value=1, max_value=120, value=20, step=1)
gender = st.selectbox("Gender", ("Male", "Female"), index=1)
polyuria = st.selectbox("Polyuria (Yes/No)", ("Yes", "No"), index=1)
polydipsia = st.selectbox("Polydipsia (Yes/No)", ("Yes", "No"), index=1)
sudden_weight_loss = st.selectbox("Sudden weight loss (Yes/No)", ("Yes", "No"), index=1)
weakness = st.selectbox("Weakness (Yes/No)", ("Yes", "No"), index=1)
polyphagia = st.selectbox("Polyphagia (Yes/No)", ("Yes", "No"), index=1)
genital_thrush = st.selectbox("Genital thrush (Yes/No)", ("Yes", "No"), index=1)
visual_blurring = st.selectbox("Visual blurring (Yes/No)", ("Yes", "No"), index=1)
itching = st.selectbox("Itching (Yes/No)", ("Yes", "No"), index=1)
irritability = st.selectbox("Irritability (Yes/No)", ("Yes", "No"), index=1)
delayed_healing = st.selectbox("Delayed healing (Yes/No)", ("Yes", "No"), index=1)
partial_paresis = st.selectbox("Partial paresis (Yes/No)", ("Yes", "No"), index=1)
muscle_stiffness = st.selectbox("Muscle stiffness (Yes/No)", ("Yes", "No"), index=1)
alopecia = st.selectbox("Alopecia (Yes/No)", ("Yes", "No"), index=1)
obesity = st.selectbox("Obesity (Yes/No)", ("Yes", "No"), index=1)

# Combine the input into a pandas DataFrame
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

# # Convert categorical columns to binary
# binary_cols = ['gender', 'polyuria', 'polydipsia', 'sudden_weight_loss', 'weakness', 
#                 'polyphagia', 'genital_thrush', 'visual_blurring', 'itching', 'irritability', 
#                 'delayed_healing', 'partial_paresis', 'muscle_stiffness', 'alopecia', 'obesity']

# for col in binary_cols:
#      le = LabelEncoder()
#      df[col] = le.fit_transform(df[col])

# scale the numerical columns
# num_cols = ['age', 'gender', 'polyuria', 'polydipsia', 'sudden_weight_loss', 'weakness', 
#             'polyphagia', 'genital_thrush', 'visual_blurring', 'itching', 'irritability', 
#             'delayed_healing', 'partial_paresis', 'muscle_stiffness', 'alopecia', 'obesity']
# df[num_cols] = scaler.fit_transform(df[num_cols])

df = scaler.transform(df)
#print(df)

# Print the contents of df
#st.write(df)

if st.button("Make Prediction"):
    # Make predictions
    prediction = model.predict(df)
    
    # Dispay the predictions
    if prediction[0] == 'Positive':
        st.write("Positive - The patient has diabetes.")
    else:
        st.write("Negative - The patient does not have diabetes.")