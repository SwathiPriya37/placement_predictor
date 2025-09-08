import streamlit as st
import pickle
import numpy as np
import os

st.title("Student Placement Predictor")

@st.cache_resource
def load_model():
    # If model is in repo root:
    model_path = "model.pkl"

    # If you host model on Google Drive / external URL, you can download it here (see notes below).
    if not os.path.exists(model_path):
        st.info("model.pkl not found locally — you may need to upload it or enable download logic.")
        return None

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

if model is None:
    st.error("Model not loaded. Make sure 'model.pkl' exists in the app folder.")
else:
    cgpa = st.number_input("Enter CGPA", min_value=0.0, max_value=10.0, value=6.8, step=0.1)
    iq = st.number_input("Enter IQ", min_value=50, max_value=200, value=120, step=1)

    if st.button("Predict"):
        X = np.array([[cgpa, iq]])
        pred = model.predict(X)
        prob = None
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X)[0,1]

        if pred[0] == 1:
            if prob is not None:
                st.success(f"Likely to get placed — probability ≈ {prob:.2f}")
            else:
                st.success("Likely to get placed")
        else:
            if prob is not None:
                st.warning(f"May not get placed — probability ≈ {prob:.2f}")
            else:
                st.warning("May not get placed")
