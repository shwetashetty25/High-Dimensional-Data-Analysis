import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

st.set_page_config(page_title="Wine Group Finder", page_icon="🍷", layout="wide")

# ---------- load saved models ----------
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
kmeans = joblib.load("kmeans.pkl")
X_pca, labels = joblib.load("pca_data.pkl")

wine = load_wine()
names = wine.feature_names

# ---------- sidebar: model info ----------
st.sidebar.title("Model details")
st.sidebar.write("**Pipeline:** StandardScaler → PCA (2 components) → K-Means (K=3)")
st.sidebar.write(f"**Variance kept by 2 PCs:** {pca.explained_variance_ratio_.sum() * 100:.1f}%")
st.sidebar.write("**Training samples:**", len(X_pca))
st.sidebar.write("**Cluster sizes:**", np.bincount(labels).tolist())
st.sidebar.info("Hierarchical clustering was used only for comparison, because it cannot predict new samples.")

# ---------- header ----------
st.title("🍷 Wine Sample Group Finder")
st.write("Enter the 13 chemical measurements of a wine sample. The app finds its group "
         "and shows where it sits in the 2D PCA space.")

# ---------- starting values ----------
choice = st.radio("Start from:", ["Dataset average", "Example: Group 0", "Example: Group 1", "Example: Group 2"],
                  horizontal=True)
if choice == "Dataset average":
    defaults = wine.data.mean(axis=0)
else:
    g = int(choice[-1])
    defaults = wine.data[labels == g].mean(axis=0)

# ---------- 13 inputs in 3 columns ----------
st.subheader("Sample measurements")
cols = st.columns(3)
values = []
for i, name in enumerate(names):
    with cols[i % 3]:
        v = st.number_input(name, value=float(round(defaults[i], 2)), key=f"{name}_{choice}")
        values.append(v)

# ---------- predict ----------
if st.button("Predict group", type="primary"):
    sample = pd.DataFrame([values], columns=names)
    sample_pca = pca.transform(scaler.transform(sample))
    group = int(kmeans.predict(sample_pca)[0])
    distance = np.linalg.norm(sample_pca - kmeans.cluster_centers_[group])

    st.divider()
    st.subheader("Result")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Assigned group", f"Group {group}")
    m2.metric("PC1", f"{sample_pca[0][0]:.2f}")
    m3.metric("PC2", f"{sample_pca[0][1]:.2f}")
    m4.metric("Distance to centre", f"{distance:.2f}")

    # distance to every cluster centre
    dists = np.linalg.norm(kmeans.cluster_centers_ - sample_pca, axis=1)
    st.write("**Distance to each group centre**")
    st.dataframe(pd.DataFrame({"Group": [0, 1, 2], "Distance": dists.round(2)}), hide_index=True)

    # plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", alpha=0.6)
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
               c="black", marker="X", s=150, label="Cluster centres")
    ax.scatter(sample_pca[0][0], sample_pca[0][1], c="red", marker="*", s=400,
               edgecolors="black", label="Your sample")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Your sample in the 2D PCA space")
    ax.legend()
    st.pyplot(fig)