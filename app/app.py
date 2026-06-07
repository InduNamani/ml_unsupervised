
import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
import plotly.express as px
import matplotlib.pyplot as plt

st.set_page_config(page_title="Hierarchical Clustering Dashboard", layout="wide")

st.markdown("""
<style>
.main{
background:linear-gradient(135deg,#eef2ff,#f8fafc);
}
.block-container{
padding-top:1rem;
}
.metric-card{
padding:10px;
border-radius:12px;
background:white;
box-shadow:0px 2px 10px rgba(0,0,0,0.1);
}
h1{
text-align:center;
color:#4338ca;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Hierarchical Clustering Analytics Dashboard")

BASE_DIR = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE_DIR/"data"/"customer_data.csv")

with st.sidebar:
    st.header("⚙ Hyperparameters")
    n_clusters = st.slider("Number of Clusters",2,6,3)
    linkage_method = st.selectbox("Linkage Method",
                                  ["ward","complete","average","single"])

st.subheader("Dataset")
st.dataframe(df, use_container_width=True)

X = StandardScaler().fit_transform(df)

model = AgglomerativeClustering(
    n_clusters=n_clusters,
    linkage=linkage_method
)

labels = model.fit_predict(X)
df["Cluster"] = labels

c1,c2,c3 = st.columns(3)
c1.metric("Rows", len(df))
c2.metric("Features", 2)
c3.metric("Clusters", n_clusters)

col1,col2 = st.columns(2)

with col1:
    fig = px.scatter(
        df,
        x="Age",
        y="Income",
        color=df["Cluster"].astype(str),
        size="Income",
        title="Cluster Visualization"
    )
    st.plotly_chart(fig,use_container_width=True)

with col2:
    counts=df["Cluster"].value_counts().sort_index()
    fig2=px.bar(
        x=counts.index.astype(str),
        y=counts.values,
        labels={"x":"Cluster","y":"Count"},
        title="Cluster Distribution"
    )
    st.plotly_chart(fig2,use_container_width=True)

st.subheader("🌳 Dendrogram")
Z = linkage(X, method=linkage_method)
fig3, ax = plt.subplots(figsize=(10,4))
dendrogram(Z, ax=ax)
st.pyplot(fig3)

st.subheader("Clustered Data")
st.dataframe(df,use_container_width=True)
