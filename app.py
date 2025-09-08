import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("🎓 Student Placement Predictor")

# Inputs
cgpa = st.number_input("Enter CGPA", min_value=0.0, max_value=10.0, step=0.1)
iq = st.number_input("Enter IQ", min_value=0.0, max_value=200.0, step=1.0)

if st.button("Predict"):
    # Rule based condition
    if cgpa <= 5:
        st.error("❌ No possibility of placement (CGPA too low)")
    else:
        # Model prediction
        X = np.array([[cgpa, iq]])
        pred = model.predict(X)
        prob = None

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X)[0, 1]

        if pred[0] == 1:
            if prob is not None:
                st.success(f"🎉 Likely to get placed — probability ≈ {prob:.2f}")
            else:
                st.success("🎉 Likely to get placed")
        else:
            if prob is not None:
                st.error(f"❌ No possibility of placement — probability ≈ {1-prob:.2f}")
            else:
                st.error("❌ No possibility of placement")
