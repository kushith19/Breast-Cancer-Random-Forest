import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import numpy as np
from src.data_preprocessing import load_and_preprocess
from src.model_training import train_models

st.title(" Breast Cancer Prediction App")


X_train, X_test, y_train, y_test, feature_names = load_and_preprocess()
models = train_models(X_train, y_train)
model = models["Random Forest"]

st.write("Enter values for the following features:")

inputs = []
for feat in feature_names[:10]:
    val = st.number_input(feat, min_value=0.00, max_value=3000.0, step=0.1)
    inputs.append(val)

if st.button("Predict"):
    prediction = model.predict([inputs + [0]*(len(feature_names)-10)]) 
    st.success("Prediction: Malignant" if prediction[0] == 0 else "Benign")
