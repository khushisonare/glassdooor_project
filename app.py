import streamlit as st
import pickle
import numpy as np

# Load the model
with open("model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
scaler = data["scaler"]
pca = data["pca"]
features = data["features"]  # 10 selected features

st.title("Glassdoor Salary 💰 Predictor")
st.write("### Check the boxes that apply to your skills and job type:")

# === Human-Friendly Inputs ===
python = st.checkbox("Python?")
excel = st.checkbox("Excel?")
aws = st.checkbox("AWS?")
sql = st.checkbox("SQL?")

hourly = st.radio("Is the job paid on an hourly basis?", ['Yes', 'No'])
hourly = 1 if hourly == 'Yes' else 0

employer_provided = st.radio("Is the salary provided directly by the employer?", ['Yes', 'No'])
employer_provided = 1 if employer_provided == 'Yes' else 0

min_salary = st.number_input("Minimum Salary ($K)", value=50)
max_salary = st.number_input("Maximum Salary ($K)", value=100)

# === Backend Feature Vector ===
feature_dict = {
    'python': int(python),
    'excel': int(excel),
    'aws': int(aws),
    'sql': int(sql),
    'hourly': hourly,
    'employer_provided': employer_provided,
    'min_salary': min_salary,
    'max_salary': max_salary,
    # Add placeholders for any other fixed selected features
}

# Match the exact order from training
input_vector = [feature_dict.get(f, 0) for f in features]
input_vector = np.array(input_vector).reshape(1, -1)

# Transform & Predict
if st.button("Predict Salary"):
    scaled_input = scaler.transform(input_vector)
    reduced_input = pca.transform(scaled_input)
    pred = model.predict(reduced_input)[0]
    st.success(f"Estimated Average Salary (USD): ${pred:.2f}")
    st.info(f"Estimated Average Salary (INR): ₹{pred * 83:.2f}")
