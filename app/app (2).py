import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="K-Means Clustering",
    page_icon="🔵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main { background: #0d0d0d; }
[data-testid="stSidebar"] { background: #111111; border-right: 1px solid #2a2a2a; }

.title-block {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    border: 1px solid #1e3a4a;
}
.title-block h1 {
    font-family: 'Space Mono', monospace;
    color: #00d4ff;
    font-size: 2.4rem;
    margin: 0;
    letter-spacing: -1px;
}
.title-block p { color: #8ab4c2; margin: 0.5rem 0 0; font-size: 1rem; }

.metric-card {
    background: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-card .val {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    color: #00d4ff;
    font-weight: 700;
}
.metric-card .lbl { color: #666; font-size: 0.8rem; margin-top: 4px; text-transform: uppercase; letter-spacing: 1px; }

.section-header {
    font-family: 'Space Mono', monospace;
    color: #00d4ff;
    font-size: 1rem;
    border-bottom: 1px solid #2a2a2a;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
    text-transform: uppercase;
    letter-spacing: 2px;
}

.info-box {
    background: #0a1929;
    border-left: 3px solid #00d4ff;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    color: #a0c4d8;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-block">
    <h1>🔵 K-Means Clustering</h1>
    <p>Unsupervised Machine Learning · Module 1 · Interactive Explorer</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    st.markdown("**Dataset**")
    dataset_source = st.radio("Source", ["Built-in Iris Dataset", "Upload CSV"], label_visibility="collapsed")

    df_raw = None
    if dataset_source == "Built-in Iris Dataset":
        import os
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'iris.csv')
        df_raw = pd.read_csv(data_path)
        st.success("✅ Iris dataset loaded (150 rows)")
        st.markdown("""
        <div class='info-box'>
        📦 <b>Iris Dataset</b><br>
        150 samples · 4 features · 3 species<br>
        Source: sklearn.datasets.load_iris
        </div>
        """, unsafe_allow_html=True)
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            df_raw = pd.read_csv(uploaded)
            st.success(f"✅ Loaded {len(df_raw)} rows")

    st.divider()
    st.markdown("**K-Means Parameters**")
    k = st.slider("Number of Clusters (K)", min_value=2, max_value=10, value=3)
    init_method = st.selectbox("Initialization Method", ["k-means++", "random"])
    max_iter = st.slider("Max Iterations", 100, 500, 300, step=50)
    n_init = st.slider("Number of Initializations (n_init)", 1, 20, 10)
    random_state = st.number_input("Random State", value=42, step=1)
    scale = st.checkbox("Standardize Features (Recommended)", value=True)

    st.divider()
    st.markdown("**Elbow Method**")
    k_range_max = st.slider("Max K to evaluate", 5, 15, 10)

# ── Main Logic ────────────────────────────────────────────────────────────────
if df_raw is None:
    st.info("👈 Select a dataset from the sidebar to begin.")
    st.stop()

# Feature selection
numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
label_col = None

# Try to detect label column
for col in df_raw.columns:
    if df_raw[col].dtype == object or (col.lower() in ['species', 'label', 'class', 'target']):
        label_col = col
        break

st.markdown('<div class="section-header">📋 Dataset Preview</div>', unsafe_allow_html=True)
col1, col2 = st.columns([2, 1])
with col1:
    st.dataframe(df_raw.head(10), use_container_width=True)
with col2:
    st.markdown("**Shape**")
    st.markdown(f"`{df_raw.shape[0]} rows × {df_raw.shape[1]} cols`")
    st.markdown("**Feature Columns**")
    st.write(numeric_cols)
    if label_col:
        st.markdown(f"**Label Column:** `{label_col}`")

# Prepare features
X = df_raw[numeric_cols].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) if scale else X.values

# ── Elbow Method ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📈 Elbow Method (Optimal K)</div>', unsafe_allow_html=True)

wcss = []
k_values = list(range(1, k_range_max + 1))
for ki in k_values:
    km = KMeans(n_clusters=ki, init=init_method, max_iter=max_iter,
                n_init=n_init, random_state=int(random_state))
    km.fit(X_scaled)
    wcss.append(km.inertia_)

fig_elbow, ax = plt.subplots(figsize=(9, 4))
fig_elbow.patch.set_facecolor('#0d0d0d')
ax.set_facecolor('#111111')
ax.plot(k_values, wcss, 'o-', color='#00d4ff', linewidth=2.5, markersize=8,
        markerfacecolor='#ff6b6b', markeredgecolor='#00d4ff', markeredgewidth=2)
ax.axvline(x=k, color='#ff6b6b', linestyle='--', linewidth=1.5, alpha=0.8, label=f'Selected K={k}')
ax.set_xlabel('Number of Clusters (K)', color='#8ab4c2', fontsize=11)
ax.set_ylabel('WCSS (Inertia)', color='#8ab4c2', fontsize=11)
ax.set_title('Elbow Method — Within Cluster Sum of Squares', color='#ffffff', fontsize=13, pad=12)
ax.tick_params(colors='#555')
ax.spines[['top', 'right']].set_visible(False)
ax.spines[['left', 'bottom']].set_color('#2a2a2a')
ax.legend(facecolor='#1a1a1a', edgecolor='#2a2a2a', labelcolor='#ccc')
ax.grid(True, color='#1e1e1e', linestyle='--')
st.pyplot(fig_elbow)

# ── Fit K-Means ───────────────────────────────────────────────────────────────
kmeans = KMeans(n_clusters=k, init=init_method, max_iter=max_iter,
                n_init=n_init, random_state=int(random_state))
labels = kmeans.fit_predict(X_scaled)
df_raw = df_raw.copy()
df_raw['Cluster'] = labels

inertia = kmeans.inertia_
sil_score = silhouette_score(X_scaled, labels) if k > 1 else 0.0
n_iter = kmeans.n_iter_

# ── Metrics ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Model Metrics</div>', unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
for col, val, lbl in zip(
    [m1, m2, m3, m4],
    [k, f"{inertia:.1f}", f"{sil_score:.4f}", n_iter],
    ["Clusters (K)", "Inertia (WCSS)", "Silhouette Score", "Iterations to Converge"]
):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="val">{val}</div>
            <div class="lbl">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("")
st.markdown("""
<div class='info-box'>
📌 <b>Silhouette Score</b> ranges from -1 to +1. Values closer to <b>+1</b> indicate well-separated, tight clusters.
Values near 0 suggest overlapping clusters, and negative values indicate misclassified points.
</div>
""", unsafe_allow_html=True)

# ── PCA Scatter Plot ──────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🌐 Cluster Visualization (PCA 2D)</div>', unsafe_allow_html=True)

pca = PCA(n_components=2, random_state=int(random_state))
X_pca = pca.fit_transform(X_scaled)
centroids_pca = pca.transform(kmeans.cluster_centers_)

palette = ['#00d4ff', '#ff6b6b', '#ffd93d', '#6bcb77', '#c77dff',
           '#ff9f1c', '#2ec4b6', '#e71d36', '#011627', '#fdffb6']

fig_pca, ax = plt.subplots(figsize=(9, 5))
fig_pca.patch.set_facecolor('#0d0d0d')
ax.set_facecolor('#111111')

for ci in range(k):
    mask = labels == ci
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=palette[ci % len(palette)], label=f'Cluster {ci}',
               alpha=0.75, s=60, edgecolors='none')

ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
           c='white', marker='*', s=300, zorder=5,
           edgecolors='#222', linewidths=0.5, label='Centroids')

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', color='#8ab4c2')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', color='#8ab4c2')
ax.set_title('K-Means Clusters — PCA Projection', color='white', fontsize=13, pad=12)
ax.tick_params(colors='#555')
ax.spines[['top', 'right']].set_visible(False)
ax.spines[['left', 'bottom']].set_color('#2a2a2a')
ax.legend(facecolor='#1a1a1a', edgecolor='#2a2a2a', labelcolor='#ccc', loc='best')
ax.grid(True, color='#1e1e1e', linestyle='--', alpha=0.5)
st.pyplot(fig_pca)

# ── Silhouette Plot ───────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔷 Silhouette Analysis</div>', unsafe_allow_html=True)

if k > 1:
    sil_vals = silhouette_samples(X_scaled, labels)
    fig_sil, ax = plt.subplots(figsize=(9, 4))
    fig_sil.patch.set_facecolor('#0d0d0d')
    ax.set_facecolor('#111111')

    y_lower = 10
    for ci in range(k):
        ci_vals = np.sort(sil_vals[labels == ci])
        size = ci_vals.shape[0]
        y_upper = y_lower + size
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ci_vals,
                         facecolor=palette[ci % len(palette)], alpha=0.8)
        ax.text(-0.05, y_lower + 0.5 * size, str(ci), color='white', fontsize=9)
        y_lower = y_upper + 10

    ax.axvline(x=sil_score, color='#ff6b6b', linestyle='--', linewidth=1.5,
               label=f'Avg Score = {sil_score:.4f}')
    ax.set_xlabel('Silhouette Coefficient', color='#8ab4c2')
    ax.set_ylabel('Cluster', color='#8ab4c2')
    ax.set_title('Silhouette Plot per Cluster', color='white', fontsize=13, pad=12)
    ax.tick_params(colors='#555')
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#2a2a2a')
    ax.legend(facecolor='#1a1a1a', edgecolor='#2a2a2a', labelcolor='#ccc')
    ax.grid(True, color='#1e1e1e', linestyle='--', alpha=0.4)
    st.pyplot(fig_sil)

# ── Pairplot ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔗 Feature Pair Plot</div>', unsafe_allow_html=True)

plot_cols = numeric_cols[:4]  # limit to 4 features for clarity
pair_df = df_raw[plot_cols + ['Cluster']].copy()
pair_df['Cluster'] = pair_df['Cluster'].astype(str)

fig_pair = plt.figure(figsize=(10, 8))
fig_pair.patch.set_facecolor('#0d0d0d')
pair_grid = sns.pairplot(pair_df, hue='Cluster', diag_kind='kde',
                          plot_kws={'alpha': 0.6, 's': 30},
                          palette={str(i): palette[i % len(palette)] for i in range(k)})
pair_grid.figure.patch.set_facecolor('#0d0d0d')
for ax in pair_grid.axes.flatten():
    if ax:
        ax.set_facecolor('#111111')
        ax.spines[['top', 'right', 'left', 'bottom']].set_color('#2a2a2a')
        ax.tick_params(colors='#555')
        ax.xaxis.label.set_color('#8ab4c2')
        ax.yaxis.label.set_color('#8ab4c2')

st.pyplot(pair_grid.figure)

# ── Cluster Summary Table ─────────────────────────────────────────────────────
st.markdown('<div class="section-header">📋 Cluster Summary Statistics</div>', unsafe_allow_html=True)
summary = df_raw.groupby('Cluster')[numeric_cols].mean().round(3)
summary['Count'] = df_raw.groupby('Cluster').size()
st.dataframe(summary, use_container_width=True)

# ── Download Results ──────────────────────────────────────────────────────────
st.markdown('<div class="section-header">💾 Download Results</div>', unsafe_allow_html=True)
csv = df_raw.to_csv(index=False).encode('utf-8')
st.download_button(
    label="⬇️ Download Clustered Dataset (CSV)",
    data=csv,
    file_name="kmeans_clustered_output.csv",
    mime="text/csv"
)

st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#333;font-size:0.8rem;font-family:Space Mono,monospace;'>"
    "K-Means Clustering · Unsupervised ML · Module 1</p>",
    unsafe_allow_html=True
)
