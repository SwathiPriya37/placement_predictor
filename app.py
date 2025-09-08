import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("🎓 Student Placement Predictor")

# Inputs
cgpa = st.number_input("Enter CGPA", min_value=0.0, max_value=10.0, step=0.1)
iq = st.number_input("Enter IQ", min_value=0.0, max_value=200.0, step=1.0)

# Function to classify IQ levels
def get_iq_level(iq_score):
    if iq_score < 90:
        return "🟥 Low IQ"
    elif 90 <= iq_score < 110:
        return "🟨 Average IQ"
    elif 110 <= iq_score < 130:
        return "🟩 High IQ"
    else:
        return "💎 Genius IQ"

if st.button("Predict"):
    # Show IQ level
    iq_level = get_iq_level(iq)
    st.info(f"🧠 IQ Level: {iq_level}")

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
