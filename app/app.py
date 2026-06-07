
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Gaussian Mixture Model Explorer", layout="wide")

st.markdown("""
<style>
.main {background-color:#f7f9fc;}
h1 {color:#1f4e79;}
.stButton>button {border-radius:10px;}
</style>
""", unsafe_allow_html=True)

st.title("Gaussian Mixture Model (GMM) Explorer")

df = pd.read_csv("data/gmm_dataset.csv")

st.sidebar.header("Hyperparameters")
n_components = st.sidebar.slider("Number of Components", 2, 5, 3)
covariance_type = st.sidebar.selectbox(
    "Covariance Type",
    ["full", "tied", "diag", "spherical"]
)
random_state = st.sidebar.number_input("Random State", value=42)

st.subheader("Dataset Preview")
st.dataframe(df)

X = StandardScaler().fit_transform(df)

gmm = GaussianMixture(
    n_components=n_components,
    covariance_type=covariance_type,
    random_state=random_state
)

clusters = gmm.fit_predict(X)
df["Cluster"] = clusters

col1, col2 = st.columns(2)

with col1:
    fig = px.scatter(
        df, x="Feature1", y="Feature2",
        color=df["Cluster"].astype(str),
        title="GMM Clustering"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    counts = df["Cluster"].value_counts().sort_index()
    fig2 = px.bar(
        x=counts.index.astype(str),
        y=counts.values,
        labels={"x":"Cluster","y":"Count"},
        title="Cluster Distribution"
    )
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Model Information")
st.write("Weights:", gmm.weights_)
st.write("Means:", gmm.means_)
st.write("Log Likelihood:", gmm.score(X))
