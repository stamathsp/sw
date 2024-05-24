import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import seaborn as sns

# Title of the application
st.title('Data Analysis Application')

# Upload data section
st.header('Data Upload')
uploaded_file = st.file_uploader("Select a file:", type=["csv", "xlsx"])
if uploaded_file is not None:
    # Load data based on file type
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)
    
    # Display a preview of the data
    st.write("Data Preview:", data.head())

    # Separate features and labels assuming the last column is the label
    features = data.iloc[:, :-1]
    labels = data.iloc[:, -1]

    # 2D Visualization Tab
    st.header('2D Visualization Tab')
    
    # PCA visualization
    st.subheader('PCA')
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)
    data['PCA1'] = pca_result[:,0]
    data['PCA2'] = pca_result[:,1]
    fig, ax = plt.subplots()
    sns.scatterplot(x='PCA1', y='PCA2', hue=labels, data=data, ax=ax)
    st.pyplot(fig)

    # t-SNE visualization
    st.subheader('t-SNE')
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(features)
    data['tSNE1'] = tsne_result[:,0]
    data['tSNE2'] = tsne_result[:,1]
    fig, ax = plt.subplots()
    sns.scatterplot(x='tSNE1', y='tSNE2', hue=labels, data=data, ax=ax)
    st.pyplot(fig)

    # Machine Learning Tabs
    st.header('Machine Learning Tabs')

    # Classification Tab
    st.subheader('Classification')
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    
    # Train logistic regression classifier
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    
    # Predict and calculate accuracy
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write('Accuracy:', acc)
    st.write('Confusion Matrix:', confusion_matrix(y_test, y_pred))

    # Clustering Tab
    st.subheader('Clustering')
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(features)
    clusters = kmeans.predict(features)
    data['Cluster'] = clusters
    
    # Visualize clusters using PCA components
    fig, ax = plt.subplots()
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=data, ax=ax)
    st.pyplot(fig)

    # Info Tab
    st.write("""
    ### Development Team
    - Member 1: Pavlos - Marios Giannakos
    - Member 2: Stamatis Petrou
    """)
