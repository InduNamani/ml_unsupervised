import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Hierarchical Clustering Dashboard", layout="wide")

# -------------------------------
# CUSTOM CSS
# -------------------------------
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    h1 {
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# TITLE
# -------------------------------
st.title("📊 Advanced Hierarchical Clustering Dashboard")

# -------------------------------
# SIDEBAR SETTINGS
# -------------------------------
st.sidebar.header("⚙️ Hyperparameters")

file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
else:
    df = pd.read_csv(r"C:\Users\indun\Downloads\customers.csv")

# Preview
st.subheader("📂 Dataset")
st.dataframe(df)

# Feature selection
features = st.sidebar.multiselect(
    "Select Features",
    df.columns[1:],
    default=df.columns[1:3]
)

if len(features) < 2:
    st.warning("Select at least 2 features")
    st.stop()

X = df[features]

# -------------------------------
# HYPERPARAMETERS
# -------------------------------
n_clusters = st.sidebar.slider("Number of Clusters", 2, 8, 3)

linkage_method = st.sidebar.selectbox(
    "Linkage Method",
    ["ward", "complete", "average", "single"]
)

distance_metric = st.sidebar.selectbox(
    "Distance Metric",
    ["euclidean", "manhattan", "cosine"]
)

scaling = st.sidebar.checkbox("Apply Standard Scaling")

# -------------------------------
# DATA SCALING
# -------------------------------
if scaling:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
else:
    X_scaled = X.values

# -------------------------------
# LAYOUT
# -------------------------------
col1, col2 = st.columns(2)

# -------------------------------
# DENDROGRAM
# -------------------------------
with col1:
    st.subheader("🌳 Dendrogram")

    try:
        linked = linkage(X_scaled, method=linkage_method)

        fig, ax = plt.subplots()
        dendrogram(linked, ax=ax)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")

# -------------------------------
# CLUSTER MODEL
# -------------------------------
with col2:
    st.subheader("📌 Cluster Visualization")

    try:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage_method,
            metric=distance_metric if linkage_method != "ward" else "euclidean"
        )

        labels = model.fit_predict(X_scaled)

        fig2, ax2 = plt.subplots()

        scatter = ax2.scatter(
            X.iloc[:, 0],
            X.iloc[:, 1],
            c=labels
        )

        ax2.set_xlabel(features[0])
        ax2.set_ylabel(features[1])

        st.pyplot(fig2)

    except Exception as e:
        st.error(f"Model Error: {e}")

# -------------------------------
# EVALUATION
# -------------------------------
st.subheader("📈 Model Evaluation")

try:
    score = silhouette_score(X_scaled, labels)
    st.success(f"Silhouette Score: {score:.3f}")
except:
    st.warning("Silhouette score not available")

# -------------------------------
# RESULT TABLE
# -------------------------------
df["Cluster"] = labels

st.subheader("📊 Clustered Data")
st.dataframe(df)

# -------------------------------
# DOWNLOAD BUTTON
# -------------------------------
csv = df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="📥 Download Clustered Data",
    data=csv,
    file_name="clustered_data.csv",
    mime="text/csv",
)

# -------------------------------
# INSIGHTS
# -------------------------------
st.subheader("🧠 Insights")

cluster_counts = df["Cluster"].value_counts()

st.write("Cluster Distribution:")
st.bar_chart(cluster_counts)