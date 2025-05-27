import numpy as np
import pickle
import streamlit as st
import os

# Resolve path to model directory (same as script location)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load all models from .sav files
model_names = ["RandomForest", "GradientBoosting", "ExtraTrees", "DecisionTree", "XGBoost"]
models = {}

for name in model_names:
    model_path = os.path.join(BASE_DIR, f"{name}_model.sav")
    try:
        with open(model_path, "rb") as f:
            models[name] = pickle.load(f)
    except FileNotFoundError:
        st.warning(f"⚠️ Model file '{name}_model.sav' not found in {BASE_DIR}")
        continue

# Prediction logic
def cardio_prediction(input_data):
    input_data_np = np.asarray(input_data).reshape(1, -1)
    
    probabilities = {}
    for name, model in models.items():
        prob = model.predict_proba(input_data_np)[0][1]
        probabilities[name] = prob

    high_probs = {k: v for k, v in probabilities.items() if v >= 0.5}
    chosen_model_name = max(high_probs, key=high_probs.get) if high_probs else min(probabilities, key=probabilities.get)

    chosen_model = models[chosen_model_name]
    final_prediction = chosen_model.predict(input_data_np)[0]
    final_probability = probabilities[chosen_model_name]

    result = "⚠️ Patient is at risk of cardiovascular disease." if final_prediction == 1 else "✅ Patient is likely healthy."

    return result, final_probability, chosen_model_name

# Streamlit UI
def main():
    st.title("🫀 Cardiovascular Disease Prediction Web App")
    st.write("Enter the following details:")

    age = st.text_input("Age (years)")
    gender = st.selectbox("Gender", [1, 2], format_func=lambda x: "Female" if x == 1 else "Male")
    height = st.text_input("Height (cm)")
    weight = st.text_input("Weight (kg)")
    ap_hi = st.text_input("Systolic BP")
    ap_lo = st.text_input("Diastolic BP")
    cholesterol = st.selectbox("Cholesterol", [1, 2, 3])
    gluc = st.selectbox("Glucose", [1, 2, 3])
    smoke = st.selectbox("Smoker", [0, 1])
    alco = st.selectbox("Alcohol consumption", [0, 1])
    active = st.selectbox("Physically active", [0, 1])

    if st.button("Predict Cardiovascular Risk"):
        try:
            input_list = [
                int(age), int(gender), int(height), int(weight),
                int(ap_hi), int(ap_lo), int(cholesterol), int(gluc),
                int(smoke), int(alco), int(active)
            ]

            result, probability, model_used = cardio_prediction(input_list)

            st.success(result)
            st.info(f"🧠 {model_used} predicted a probability of {probability * 100:.2f}%")
        except Exception as e:
            st.error(f"❌ Please make sure all inputs are valid numeric values.\n\n{str(e)}")

if __name__ == '__main__':
    main()
