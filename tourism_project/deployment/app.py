import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="jyotisharma/tourism_package_prediction_model", filename="tourism_package_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Prediction
st.title("Tourism Package Prediction App")
st.write("""
This application is for predicting the likelihood of purchasing the Wellness Tourism Package.
Please fill in the information below:
""")

# User input
age = st.number_input("Age", min_value=18, max_value=80, value=30)
typeof_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
gender = st.selectbox("Gender", ["Male", "Female"])
product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
passport = st.selectbox("Has Passport?", [0, 1])
own_car = st.selectbox("Owns a Car?", [0, 1])
preferred_star = st.selectbox("Preferred Property Star", [3, 4, 5])
num_children = st.selectbox("Number of Children Visiting", [0, 1, 2, 3])
num_persons = st.selectbox("Number of Persons Visiting", [1, 2, 3, 4, 5])
num_followups = st.selectbox("Number of Follow-ups", [1, 2, 3, 4, 5, 6])
duration_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=150, value=15)
num_trips = st.number_input("Number of Trips", min_value=1, max_value=30, value=3)
pitch_score = st.selectbox("Pitch Satisfaction Score (1–5)", [1, 2, 3, 4, 5])
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=200000, value=25000)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": typeof_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_persons,
    "NumberOfFollowups": num_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_score,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": num_children,
    "Designation": designation,
    "MonthlyIncome": monthly_income
}])


if st.button("Predict Package Purchasing"):
    prediction = model.predict(input_data)[0]
    result = "Package Purchase" if prediction == 1 else "No Purchase"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
