# app.py
import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("data/sleep_health_and_lifestyle_dataset.csv")
    df.dropna(inplace=True)

    # Create derived features
    df["Sleep Efficiency"] = (df["Sleep Duration"] / 8) * 100

    # Create Heart Disease label based on logic
    df["Heart Disease"] = (
        (df["BMI Category"].isin(["Obese", "Overweight"])) &
        (df["Stress Level"] >= 7) &
        (df["Sleep Disorder"].isin(["Sleep Apnea", "Insomnia"])) &
        (df["Heart Rate"] >= 80)
    ).astype(int)
    
    return df

df = load_data()

# Define features
numeric = ["Sleep Duration", "Quality of Sleep", "Stress Level", "Physical Activity Level", "Heart Rate"]
categorical = ["Gender", "Occupation", "Sleep Disorder", "BMI Category"]
X = df[numeric + categorical]
y = df["Heart Disease"]

# Build model pipeline
preprocessor = ColumnTransformer([
    ("num", MinMaxScaler(), numeric),
    ("cat", OneHotEncoder(), categorical)
])
model = Pipeline([
    ("prep", preprocessor),
    ("clf", XGBClassifier(use_label_encoder=False, eval_metric="logloss"))
])
model.fit(X, y)

# Streamlit UI
st.title("❤️ Heart Disease Risk Prediction")
st.markdown("Enter your health and lifestyle details:")

# User Inputs
gender = st.selectbox("Gender", df["Gender"].unique())
occupation = st.selectbox("Occupation", df["Occupation"].unique())
sleep_disorder = st.selectbox("Sleep Disorder", df["Sleep Disorder"].unique())
sleep_duration = st.slider("Sleep Duration (hours)", 0.0, 12.0, 7.0)
quality = st.slider("Quality of Sleep (1-10)", 1, 10, 6)
stress = st.slider("Stress Level (1-10)", 1, 10, 5)
activity = st.slider("Physical Activity Level (min/day)", 0, 120, 30)
bmi_category = st.selectbox("BMI Category", df["BMI Category"].unique())
heart_rate = st.slider("Heart Rate (bpm)", 40, 120, 70)

# Assemble input
user_input = pd.DataFrame([{
    "Gender": gender,
    "Occupation": occupation,
    "Sleep Disorder": sleep_disorder,
    "Sleep Duration": sleep_duration,
    "Quality of Sleep": quality,
    "Stress Level": stress,
    "Physical Activity Level": activity,
    "BMI Category": bmi_category,
    "Heart Rate": heart_rate
}])

# Make prediction
prediction = model.predict(user_input)[0]
probability = model.predict_proba(user_input)[0][1]

# Display result
st.subheader("Prediction Result:")
st.write("🧠 **Heart Disease Risk:**", "High" if prediction == 1 else "Low")
st.write("📊 **Confidence:**", f"{probability:.2%}")



















# import streamlit as st

# st.title("Hello Streamlit!")
# st.write("If you see this, your app is working.")

# import streamlit as st
# import pandas as pd
# from xgboost import XGBClassifier
# from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline

# # Load and preprocess data
# @st.cache_data
# def load_data():
#     df = pd.read_csv("data/sleep_health_and_lifestyle_dataset.csv")
#     df.dropna(inplace=True)
#     df["Sleep Efficiency"] = (df["Sleep Duration"] / 8) * 100
#     return df

# df = load_data()

# # Define features
# numeric = ["Sleep Duration", "Quality of Sleep", "Stress Level", "Physical Activity Level", "BMI Category", "Heart Rate"]
# categorical = ["Gender", "Occupation", "Sleep Disorder"]
# X = df[numeric + categorical]
# y = df["Heart Disease"]

# # Build model pipeline
# preprocessor = ColumnTransformer([
#     ("num", MinMaxScaler(), numeric),
#     ("cat", OneHotEncoder(), categorical)
# ])
# model = Pipeline([
#     ("prep", preprocessor),
#     ("clf", XGBClassifier(use_label_encoder=False, eval_metric="logloss"))
# ])
# model.fit(X, y)

# # Streamlit UI
# st.title("❤️ Heart Disease Risk Prediction")
# st.markdown("Enter your health and lifestyle details:")

# # User Inputs
# gender = st.selectbox("Gender", df["Gender"].unique())
# occupation = st.selectbox("Occupation", df["Occupation"].unique())
# sleep_disorder = st.selectbox("Sleep Disorder", df["Sleep Disorder"].unique())
# sleep_duration = st.slider("Sleep Duration (hours)", 0.0, 12.0, 7.0)
# quality = st.slider("Quality of Sleep (1-10)", 1, 10, 6)
# stress = st.slider("Stress Level (1-10)", 1, 10, 5)
# activity = st.slider("Physical Activity Level (min/day)", 0, 120, 30)
# bmi = st.slider("BMI", 10.0, 50.0, 25.0)
# heart_rate = st.slider("Heart Rate (bpm)", 40, 120, 70)

# # Assemble input
# user_input = pd.DataFrame([{
#     "Gender": gender,
#     "Occupation": occupation,
#     "Sleep Disorder": sleep_disorder,
#     "Sleep Duration": sleep_duration,
#     "Quality of Sleep": quality,
#     "Stress Level": stress,
#     "Physical Activity Level": activity,
#     "BMI": bmi,
#     "Heart Rate": heart_rate
# }])

# # Make prediction
# prediction = model.predict(user_input)[0]
# probability = model.predict_proba(user_input)[0][1]

# # Display result
# st.subheader("Prediction Result:")
# st.write("🧠 **Heart Disease Risk:**", "High" if prediction == 1 else "Low")
# st.write("📊 **Confidence:**", f"{probability:.2%}")

