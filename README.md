# High-Dimensional Data Analysis Using PCA and Clustering

## Case Study 48 — Machine Learning

### Project Overview

This project focuses on **High-Dimensional Data Analysis** using Principal Component Analysis (PCA) and clustering techniques. The dataset contains multiple features, making direct analysis and visualization difficult. PCA is used to reduce the dimensionality of the data while retaining the most important information. The reduced data is then analyzed using clustering techniques to identify meaningful groups within the dataset.

The project includes data preprocessing, exploratory data analysis, feature scaling, PCA, K-Means Clustering, Hierarchical Clustering, cluster evaluation, and visualization. An interactive **Streamlit application** is also developed to present the analysis and results.

## Problem Statement

High-dimensional datasets often contain a large number of features, making them difficult to visualize, interpret, and analyze effectively. The objective of this project is to reduce the dimensionality of the dataset using PCA and identify natural groupings in the reduced data using clustering algorithms.

## Objectives

* Perform exploratory data analysis on the dataset.
* Preprocess and scale the features.
* Apply PCA for dimensionality reduction.
* Analyze the variance explained by the principal components.
* Apply K-Means Clustering to the reduced data.
* Determine a suitable number of clusters using the Elbow Method.
* Apply Hierarchical Clustering and visualize the results using a Dendrogram.
* Evaluate the quality of the clusters using appropriate metrics.
* Present the analysis through an interactive Streamlit application.

## Machine Learning Techniques

### Principal Component Analysis (PCA)

PCA is a dimensionality reduction technique that transforms the original features into a smaller set of principal components. These components retain the maximum possible variance from the original dataset while reducing the number of dimensions.

### K-Means Clustering

K-Means is an unsupervised learning algorithm that divides data points into a specified number of clusters based on their similarity and distance from cluster centroids.

### Hierarchical Clustering

Hierarchical Clustering creates a hierarchy of clusters by progressively combining similar data points or groups. A **Dendrogram** is used to visualize this hierarchical structure.

### Elbow Method

The Elbow Method is used to identify a suitable number of clusters for K-Means by analyzing the change in within-cluster variation for different values of K.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Streamlit
* Jupyter Notebook

## Project Structure

```text
high-dimensional-data-analysis-pca-clustering/
│
├── app.py
├── case_study_48_pca_clustering.ipynb
├── kmeans.pkl
├── pca.pkl
├── pca_data.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
```

### File Description

| File                                 | Description                                             |
| ------------------------------------ | ------------------------------------------------------- |
| `app.py`                             | Streamlit application for the project                   |
| `case_study_48_pca_clustering.ipynb` | Notebook containing data analysis and ML implementation |
| `kmeans.pkl`                         | Saved K-Means clustering model                          |
| `pca.pkl`                            | Saved PCA model                                         |
| `pca_data.pkl`                       | PCA-transformed dataset                                 |
| `scaler.pkl`                         | Saved feature scaling model                             |
| `requirements.txt`                   | Required Python libraries                               |

## How to Run the Project

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/YOUR_USERNAME/high-dimensional-data-analysis-pca-clustering.git
cd high-dimensional-data-analysis-pca-clustering
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application can then be accessed through the local URL provided by Streamlit.

## Conclusion

This project demonstrates how **PCA and clustering techniques** can be used together to analyze high-dimensional data. PCA simplifies the dataset by reducing its dimensions, while K-Means and Hierarchical Clustering help identify patterns and groups within the reduced data. The Streamlit application provides an interactive way to explore and present the results.

