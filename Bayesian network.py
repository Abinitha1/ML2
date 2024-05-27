!pip install streamlit scikit-learn matplotlib

# Import required libraries
import streamlit as st
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the Iris dataset
dataset = load_iris()

# Preprocessing
X = pd.DataFrame(dataset.data, columns=['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width'])
y = pd.DataFrame(dataset.target, columns=['Targets'])
colormap = np.array(['red', 'lime', 'black'])

# Streamlit app
st.title("Iris Flower Clustering")

# Plotting
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# Real Plot
ax[0].scatter(X['Petal_Length'], X['Petal_Width'], c=colormap[y['Targets']], s=40)
ax[0].set_title('Real')

# KMeans Plot
model = KMeans(n_clusters=3)
model.fit(X)
predY = np.choose(model.labels_, [0, 1, 2]).astype(np.int64)
ax[1].scatter(X['Petal_Length'], X['Petal_Width'], c=colormap[predY], s=40)
ax[1].set_title('KMeans')

# Gaussian Mixture Model (GMM) Plot
scaler = preprocessing.StandardScaler()
xsa = scaler.fit_transform(X)
xs = pd.DataFrame(xsa, columns=X.columns)
gmm = GaussianMixture(n_components=3)
gmm.fit(xs)
y_cluster_gmm = gmm.predict(xs)
ax[2].scatter(X['Petal_Length'], X['Petal_Width'], c=colormap[y_cluster_gmm], s=40)
ax[2].set_title('GMM Classification')

# Streamlit display
st.pyplot(fig)
