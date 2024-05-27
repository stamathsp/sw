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
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import DBSCAN
import seaborn as sns
from plantuml import PlantUML

# Title of the application
st.title('F1 Data Analysis Application')

# Upload data section
st.header('Data Upload')
uploaded_file = st.file_uploader("Select a file:", type=["csv", "xlsx"])
if uploaded_file is not None:
    try:
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

        # EDA Tab
        st.header('Exploratory Data Analysis (EDA)')
        st.subheader('Data Description')
        st.write(data.describe())

        st.subheader('Correlation Matrix')
        fig, ax = plt.subplots(figsize=(12, 8))  # Increase the figure size
        sns.heatmap(data.corr(), annot=True, fmt=".1f", cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        
        st.subheader('Pairplot')
        fig = sns.pairplot(data)
        st.pyplot(fig.fig)  # Pass the PairGrid figure to st.pyplot()

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
        
        # Logistic Regression
        st.write("Logistic Regression")
        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.write('Accuracy:', acc)
        st.write('Confusion Matrix:', confusion_matrix(y_test, y_pred))

        # Random Forest Classifier
        st.write("Random Forest Classifier")
        rf_clf = RandomForestClassifier()
        rf_clf.fit(X_train, y_train)
        y_pred_rf = rf_clf.predict(X_test)
        acc_rf = accuracy_score(y_test, y_pred_rf)
        st.write('Accuracy:', acc_rf)
        st.write('Confusion Matrix:', confusion_matrix(y_test, y_pred_rf))

        # Clustering Tab
        st.subheader('Clustering')
        
        # KMeans clustering
        st.write("KMeans Clustering")
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(features)
        clusters = kmeans.predict(features)
        data['Cluster'] = clusters
        fig, ax = plt.subplots()
        sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=data, ax=ax)
        st.pyplot(fig)

        # DBSCAN clustering
        st.write("DBSCAN Clustering")
        dbscan = DBSCAN()
        clusters_dbscan = dbscan.fit_predict(features)
        data['DBSCAN_Cluster'] = clusters_dbscan
        fig, ax = plt.subplots()
        sns.scatterplot(x='PCA1', y='PCA2', hue='DBSCAN_Cluster', data=data, ax=ax)
        st.pyplot(fig)

        # Info Tab
        st.write("""
        ### Development Team
        - Μέλος 1: Παύλος - Μάριος Γιαννάκος
        - Μέλος 2: Σταμάτης Πέτρου
        """)

        st.write("""
        ### Docker Instructions
        - To build the Docker image: `docker build -t data-analysis-sw .`
        - To run the Docker container: `docker run -p 8501:8501 my-streamlit-sw`
        """)

        st.write("""
        ### GitHub Repository
        - Find the source code and collaborate on [GitHub](https://github.com/stamathsp/sw)
        """)

        # UML Diagram Section
        st.header('UML Diagram')
        plantuml_code = """
        @startuml
        class "streamlit" as streamlit {
            file_uploader(input)
        }

        class "pandas" as pandas {
            read_csv()
            read_excel()
        }

        class "sklearn.decomposition" as sklearn_decomposition {
            PCA()
        }

        class "sklearn.manifold" as sklearn_manifold {
            TSNE()
        }

        class "sklearn.model_selection" as sklearn_model_selection {
            train_test_split()
        }

        class "sklearn.metrics" as sklearn_metrics {
            accuracy_score()
            confusion_matrix()
        }

        class "sklearn.linear_model" as sklearn_linear_model {
            LogisticRegression()
        }

        class "sklearn.cluster" as sklearn_cluster {
            KMeans()
            DBSCAN()
        }

        class "sklearn.ensemble" as sklearn_ensemble {
            RandomForestClassifier()
        }

        class "seaborn" as seaborn {
            heatmap()
            pairplot()
            scatterplot()
        }

        streamlit --* pandas : uses
        streamlit --* sklearn_decomposition : uses
        streamlit --* sklearn_manifold : uses
        streamlit --* sklearn_model_selection : uses
        streamlit --* sklearn_metrics : uses
        streamlit --* sklearn_linear_model : uses
        streamlit --* sklearn_cluster : uses
        streamlit --* sklearn_ensemble : uses
        streamlit --* seaborn : uses
        @enduml
        """

        # Render UML diagram
        plantuml_server = PlantUML(url='http://www.plantuml.com/plantuml/img/')
        uml_image = plantuml_server.processes(plantuml_code)
        st.image(uml_image, caption='UML Diagram')

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.write("Please upload a file to proceed.")
