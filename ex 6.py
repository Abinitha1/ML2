# Install pgmpy and streamlit
import streamlit as st
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Load the dataset
heart_disease = pd.read_csv("/content/data7_heart.csv")

# Define the Bayesian network structure
model = BayesianModel([
    ('age', 'trestbps'),
    ('age', 'fbs'),
    ('sex', 'trestbps'),
    ('exang', 'trestbps'),
    ('trestbps', 'heartdisease'),
    ('fbs', 'heartdisease'),
    ('heartdisease', 'restecg'),
    ('heartdisease', 'thalach'),
    ('heartdisease', 'chol')
])

# Fit the model using Maximum Likelihood Estimator
model.fit(heart_disease, estimator=MaximumLikelihoodEstimator)

# Create an inference object
HeartDisease_infer = VariableElimination(model)

# Streamlit app
def main():
    st.title("Heart Disease Prediction")

    # User inputs
    age = st.number_input('Enter age', value=0, step=1)
    sex = st.selectbox('Select sex', ['Male', 'Female'])
    trestbps = st.number_input('Enter resting blood pressure', value=0, step=1)
    fbs = st.selectbox('Select fasting blood sugar level', ['> 120 mg/dl', '< 120 mg/dl'])

    # Convert categorical inputs to numerical
    sex = 0 if sex == 'Male' else 1
    fbs = 1 if fbs == '> 120 mg/dl' else 0

    # Perform inference
    q = HeartDisease_infer.query(variables=['heartdisease'], evidence={'age': age, 'sex': sex, 'trestbps': trestbps, 'fbs': fbs})

    # Output prediction
    st.write("Probability of Heart Disease:", q.values[1])

if __name__ == "__main__":
    main()
