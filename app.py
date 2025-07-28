from pycaret.classification import *
import streamlit as st
import pandas as pd

model = load_model("tuned_rf_diabetes")


def predict_diabetes(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df.iloc[0]["prediction_label"]
    return predictions


def run():
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
    glucose = st.number_input("Glucose", min_value=0, max_value=200, value=0)
    blood_pressure = st.number_input(
        "Blood Pressure", min_value=0, max_value=150, value=0
    )
    skin_thickness = st.number_input(
        "Skin Thickness", min_value=0, max_value=100, value=0
    )
    insulin = st.number_input("Insulin", min_value=0, max_value=300, value=0)
    bmi = st.number_input("BMI", min_value=0.0, max_value=50.0, value=0.0)
    age = st.number_input("Age", min_value=0, max_value=120, value=0)
    diabetes_pedigree_function = st.number_input(
        "Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.0
    )
    output = st.empty()

    input_dict = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "Age": age,
        "DiabetesPedigreeFunction": diabetes_pedigree_function,
    }
    input_data = pd.DataFrame([input_dict])
    if st.button("Predict"):
        with output.container():
            prediction = predict_diabetes(model, input_data)
            if prediction == 1:
                st.success("The model predicts that you have diabetes.")
            else:
                st.success("The model predicts that you do not have diabetes.")


if __name__ == "__main__":
    st.title("Diabetes Prediction App")
    st.write(
        "This app uses a machine learning model to predict whether a person has diabetes based on various health metrics."
    )
    run()
