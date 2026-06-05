import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings("ignore")
import os

st.set_page_config(page_title="Hierarchical Clustering", page_icon="🌿", layout="wide")

st.title("🌿 Hierarchical Clustering - Iris Dataset")
st.write("Agglomerative Hierarchical Clustering | Module 1 - Clustering Algorithms")
st.divider()

# sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    dataset_source = st.radio("Dataset", ["Built-in Iris Dataset", "Upload CSV"])

    df_raw = None
    if dataset_source == "Built-in Iris Dataset":
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'iris.csv')
        df_raw = pd.read_csv(data_path)
        st.success("Iris dataset loaded (150 rows)")
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            df_raw = pd.read_csv(uploaded)
            st.success(f"Loaded {len(df_raw)} rows")

    st.divider()
    st.subheader("Parameters")
    n_clusters = st.slider("Number of Clusters", 2, 10, 3)
    linkage_method = st.selectbox("Linkage Method", ["ward", "complete", "average", "single"])
    scale = st.checkbox("Standardize Features", value=True)

if df_raw is None:
    st.info("👈 Select a dataset from the sidebar to get started.")
    st.stop()

# features
numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()

st.subheader("📋 Dataset Preview")
col1, col2 = st.columns([2, 1])
with col1:
    st.dataframe(df_raw.head(10), use_container_width=True)
with col2:
    st.write(f"**Shape:** {df_raw.shape}")
    st.write(f"**Features:** {numeric_cols}")

# preprocess
X = df_raw[numeric_cols].dropna().values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) if scale else X

# dendrogram
st.subheader("🌳 Dendrogram")
st.write("The longest vertical line that doesn't get cut tells us the best number of clusters.")
linked = linkage(X_scaled, method=linkage_method)
fig_dend, ax = plt.subplots(figsize=(10, 4))
dendrogram(linked, truncate_mode='lastp', p=20, ax=ax)
ax.set_title(f"Dendrogram (linkage={linkage_method})")
ax.set_xlabel("Samples")
ax.set_ylabel("Distance")
st.pyplot(fig_dend)

# fit model
model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
labels = model.fit_predict(X_scaled)
df_raw = df_raw.copy()
df_raw['Cluster'] = labels

# metrics
sil = silhouette_score(X_scaled, labels)
st.subheader("📊 Metrics")
m1, m2, m3 = st.columns(3)
m1.metric("Clusters", n_clusters)
m2.metric("Silhouette Score", round(sil, 4))
m3.metric("Linkage Method", linkage_method)

# PCA plot
st.subheader("🔵 Cluster Visualization (PCA 2D)")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

fig_pca, ax = plt.subplots(figsize=(8, 5))
colors = ['#e63946', '#457b9d', '#2a9d8f', '#e9c46a', '#f4a261']
for i in range(n_clusters):
    mask = labels == i
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[i % len(colors)],
               label=f'Cluster {i}', alpha=0.7, s=60)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
ax.set_title("Hierarchical Clustering - PCA Projection")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig_pca)

# cluster counts
st.subheader("📋 Cluster Summary")
col1, col2 = st.columns(2)
with col1:
    st.write("**Cluster Sizes**")
    st.dataframe(df_raw['Cluster'].value_counts().reset_index().rename(
        columns={'index': 'Cluster', 'Cluster': 'Count'}), use_container_width=True)
with col2:
    st.write("**Mean Values per Cluster**")
    st.dataframe(df_raw.groupby('Cluster')[numeric_cols].mean().round(2), use_container_width=True)

# download
st.divider()
csv = df_raw.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download Clustered Data", csv, "hierarchical_output.csv", "text/csv")