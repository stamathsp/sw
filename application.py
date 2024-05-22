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
st.title('Εφαρμογή για Ανάλυση Δεδομένων')

# Upload data section
st.header('Φόρτωση Δεδομένων')
uploaded_file = st.file_uploader("Select a file:", type=["csv", "xlsx"])
if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)
    st.write("Προεπισκόπηση δεδομένων:", data.head())

    # Check if the last column is the label
    features = data.iloc[:, :-1]
    labels = data.iloc[:, -1]

    # 2D Visualization Tab
    st.header('2D Visualization Tab')
    st.subheader('PCA')
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)
    data['PCA1'] = pca_result[:,0]
    data['PCA2'] = pca_result[:,1]
    fig, ax = plt.subplots()
    sns.scatterplot(x='PCA1', y='PCA2', hue=labels, data=data, ax=ax)
    st.pyplot(fig)

    st.subheader('t-SNE')
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(features)
    data['tSNE1'] = tsne_result[:,0]
    data['tSNE2'] = tsne_result[:,1]
    fig, ax = plt.subplots()
    sns.scatterplot(x='tSNE1', y='tSNE2', hue=labels, data=data, ax=ax)
    st.pyplot(fig)

    # Machine Learning Tabs
    st.header('Tabs Μηχανικής Μάθησης')

    # Classification Tab
    st.subheader('Classification')
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write('Accuracy:', acc)
    st.write('Confusion Matrix:', confusion_matrix(y_test, y_pred))

    # Clustering Tab
    st.subheader('Clustering')
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(features)
    clusters = kmeans.predict(features)
    data['Cluster'] = clusters
    fig, ax = plt.subplots()
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=data, ax=ax)
    st.pyplot(fig)

    # Info Tab
    st.header('Info Tab')
    st.write("""
    ### Πληροφορίες Εφαρμογής
    Αυτή η εφαρμογή δημιουργήθηκε για την ανάλυση δεδομένων χρησιμοποιώντας αλγορίθμους μηχανικής μάθησης.
    
    ### Ομάδα Ανάπτυξης
    - Μέλος 1: Παύλος - Μάριος Γιαννάκος
    - Μέλος 2: Σταμάτης Πέτρου
    """)

# Για την διανομή μέσω Docker και GitHub, δημιουργήστε ένα Dockerfile και ανεβάστε τον κώδικα στο GitHub
