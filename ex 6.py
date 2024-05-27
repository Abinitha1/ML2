import streamlit as st
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Define the structure of the Bayesian Network
model = BayesianNetwork([
    ('A', 'C'),
    ('B', 'C')
])

# Define the Conditional Probability Distributions (CPDs)
cpd_A = TabularCPD(variable='A', variable_card=2, values=[[0.8], [0.2]])
cpd_B = TabularCPD(variable='B', variable_card=2, values=[[0.7], [0.3]])
cpd_C = TabularCPD(variable='C', variable_card=2, 
                   values=[[0.9, 0.6, 0.7, 0.1],
                           [0.1, 0.4, 0.3, 0.9]],
                   evidence=['A', 'B'],
                   evidence_card=[2, 2])

# Add CPDs to the model
model.add_cpds(cpd_A, cpd_B, cpd_C)

# Validate the model
model.check_model()

# Perform inference
inference = VariableElimination(model)

st.title("Bayesian Network Inference")

st.write("This application performs inference on a simple Bayesian Network.")

# User inputs
a = st.selectbox('Select value for A', ['True', 'False'])
b = st.selectbox('Select value for B', ['True', 'False'])

# Map user input to network values
evidence = {
    'A': 1 if a == 'True' else 0,
    'B': 1 if b == 'True' else 0
}

# Perform inference
result = inference.query(variables=['C'], evidence=evidence)

# Display the results
st.write(f"Probability of C given A={a} and B={b}:")
st.write(result)
