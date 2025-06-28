import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load training data and train model
@st.cache_data
def load_model_and_data():
    df = pd.read_csv('C:/Users/Mudassir/Downloads/Doctor/Training.csv')
    X = df.drop(['prognosis'], axis=1)
    y = df['prognosis']

    # Encode target classes
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_encoded)

    return model, X.columns.tolist(), le

# Load model, symptom list, and label encoder
model, symptom_list, label_encoder = load_model_and_data()

# Streamlit UI
st.set_page_config(page_title="Disease Predictor", layout="centered")
st.title("🩺 AI-Powered Disease Predictor")
st.markdown("Start typing your symptoms. The system will suggest matches.")

# Multiselect input with autocomplete
selected_symptoms = st.multiselect(
    "Enter Symptoms:",
    options=symptom_list,
    help="Start typing and select symptoms from the list."
)

# Convert selected symptoms to binary feature vector
input_vector = np.zeros(len(symptom_list))
for i, symptom in enumerate(symptom_list):
    if symptom in selected_symptoms:
        input_vector[i] = 1

# Prediction
if st.button("Predict Disease"):
    input_df = pd.DataFrame([input_vector], columns=symptom_list)
    prediction = model.predict(input_df)[0]
    disease = label_encoder.inverse_transform([prediction])[0]
    st.success(f"### 🧬 Predicted Disease: {disease}")
