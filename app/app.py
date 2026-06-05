import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore")
import os

st.set_page_config(page_title="Dimensionality Reduction", page_icon="📐", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background: #f5f5f0;
    color: #1a1a1a;
}
[data-testid="stSidebar"] {
    background: #1a1a2e;
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

.hero {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    background: #1a1a2e;
    color: white;
    padding: 2.5rem 3rem;
    border-radius: 12px;
    margin-bottom: 2rem;
}
.hero-left h1 { font-size: 2.2rem; font-weight: 700; margin: 0; color: white; }
.hero-left p { margin: 0.4rem 0 0; color: #a0a0c0; font-size: 0.9rem; }
.hero-right { text-align: right; }
.hero-right .big { font-size: 3rem; font-weight: 700; color: #7c83fd; line-height: 1; }
.hero-right .small { font-size: 0.75rem; color: #a0a0c0; text-transform: uppercase; letter-spacing: 1px; }

.tab-header {
    font-size: 1.1rem;
    font-weight: 600;
    padding: 0.5rem 0;
    border-bottom: 3px solid #7c83fd;
    display: inline-block;
    margin-bottom: 1rem;
    color: #1a1a2e;
}

.kpi {
    background: white;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    border-left: 4px solid #7c83fd;
}
.kpi .val { font-size: 1.8rem; font-weight: 700; color: #1a1a2e; }
.kpi .lab { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-top: 2px; }

.note {
    background: #eef0ff;
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    font-size: 0.85rem;
    color: #4a4a8a;
    margin: 0.8rem 0;
}
</style>
""", unsafe_allow_html=True)

# hero
st.markdown("""
<div class="hero">
    <div class="hero-left">
        <h1>📐 Dimensionality Reduction</h1>
        <p>PCA &nbsp;+&nbsp; t-SNE &nbsp;·&nbsp; Digits Dataset &nbsp;·&nbsp; Module 2</p>
    </div>
    <div class="hero-right">
        <div class="big">64→2</div>
        <div class="small">features reduced</div>
    </div>
</div>
""", unsafe_allow_html=True)

# sidebar
with st.sidebar:
    st.markdown("### Dataset")
    source = st.radio("", ["Built-in Digits Dataset", "Upload CSV"], label_visibility="collapsed")

    df_raw = None
    X, y = None, None

    if source == "Built-in Digits Dataset":
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'digits.csv')
        df_raw = pd.read_csv(data_path)
        digits = load_digits()
        X = digits.data
        y = digits.target
        st.success("1797 samples · 64 features · 10 classes")
    else:
        up = st.file_uploader("Upload CSV (last col = label)", type=["csv"])
        if up:
            df_raw = pd.read_csv(up)
            X = df_raw.iloc[:, :-1].values
            y = df_raw.iloc[:, -1].values
            st.success(f"{df_raw.shape[0]} samples loaded")

    st.markdown("---")
    st.markdown("### PCA Settings")
    pca_components = st.slider("PCA Components (2D viz)", 2, 2, 2)
    show_variance = st.checkbox("Show Variance Plot", value=True)

    st.markdown("---")
    st.markdown("### t-SNE Settings")
    perplexity = st.slider("Perplexity", 5, 50, 30)
    n_iter = st.select_slider("Iterations", [250, 500, 1000], value=1000)
    pca_pre = st.slider("PCA pre-reduction (before t-SNE)", 10, 50, 30)

    st.markdown("---")
    scale = st.checkbox("StandardScaler", value=True)

if X is None:
    st.info("Pick a dataset from the sidebar.")
    st.stop()

# preprocess
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) if scale else X

# dataset preview
st.markdown("### Dataset Preview")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f'<div class="kpi"><div class="val">{X.shape[0]}</div><div class="lab">Samples</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="kpi"><div class="val">{X.shape[1]}</div><div class="lab">Original Features</div></div>', unsafe_allow_html=True)
with c3:
    n_classes = len(np.unique(y))
    st.markdown(f'<div class="kpi"><div class="val">{n_classes}</div><div class="lab">Classes</div></div>', unsafe_allow_html=True)

st.markdown("")
if df_raw is not None:
    st.dataframe(df_raw.head(5), use_container_width=True)

st.divider()

# ── PCA SECTION ───────────────────────────────────────────────────────────────
st.markdown('<div class="tab-header">① Principal Component Analysis (PCA)</div>', unsafe_allow_html=True)
st.markdown("""
<div class="note">
PCA is a <b>linear</b> technique. It finds new axes (principal components) that capture the most variance in the data.
Fast and interpretable but may miss non-linear structure.
</div>
""", unsafe_allow_html=True)

pca_full = PCA(random_state=42)
pca_full.fit(X_scaled)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_95 = int(np.argmax(cumvar >= 0.95)) + 1

if show_variance:
    fig_var, ax = plt.subplots(figsize=(8, 3.5))
    fig_var.patch.set_facecolor('#f5f5f0')
    ax.set_facecolor('white')
    ax.plot(cumvar, color='#7c83fd', linewidth=2)
    ax.fill_between(range(len(cumvar)), cumvar, alpha=0.1, color='#7c83fd')
    ax.axhline(0.95, color='#ff6b6b', linestyle='--', linewidth=1.2, label=f'95% → {n_95} components')
    ax.set_xlabel('Number of Components', color='#555')
    ax.set_ylabel('Cumulative Variance', color='#555')
    ax.set_title('Explained Variance Ratio', color='#1a1a2e')
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#ddd')
    ax.tick_params(colors='#aaa')
    ax.legend(frameon=False)
    ax.grid(True, color='#eee', linestyle='--')
    st.pyplot(fig_var)

pca_2d = PCA(n_components=2, random_state=42)
X_pca = pca_2d.fit_transform(X_scaled)
var_explained = sum(pca_2d.explained_variance_ratio_) * 100

c1, c2 = st.columns(2)
with c1:
    st.markdown(f'<div class="kpi"><div class="val">{n_95}</div><div class="lab">Components for 95% variance</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="kpi"><div class="val">{var_explained:.1f}%</div><div class="lab">Variance in 2 components</div></div>', unsafe_allow_html=True)

st.markdown("")

fig_pca, ax = plt.subplots(figsize=(9, 5.5))
fig_pca.patch.set_facecolor('#f5f5f0')
ax.set_facecolor('white')
sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.7, s=25)
plt.colorbar(sc, ax=ax, label='Class Label')
ax.set_xlabel(f'PC1  ({pca_2d.explained_variance_ratio_[0]*100:.1f}% variance)', color='#555')
ax.set_ylabel(f'PC2  ({pca_2d.explained_variance_ratio_[1]*100:.1f}% variance)', color='#555')
ax.set_title('PCA — 2D Projection', color='#1a1a2e', fontsize=13)
ax.spines[['top','right']].set_visible(False)
ax.spines[['left','bottom']].set_color('#ddd')
ax.tick_params(colors='#aaa')
ax.grid(True, color='#eee', linestyle='--')
st.pyplot(fig_pca)

st.divider()

# ── t-SNE SECTION ─────────────────────────────────────────────────────────────
st.markdown('<div class="tab-header">② t-SNE (t-Distributed Stochastic Neighbor Embedding)</div>', unsafe_allow_html=True)
st.markdown("""
<div class="note">
t-SNE is a <b>non-linear</b> technique great for visualization. It preserves local structure.
Slower than PCA — we first reduce with PCA to speed it up. Not suitable for new data transformation.
</div>
""", unsafe_allow_html=True)

with st.spinner("Running t-SNE... this takes a moment ⏳"):
    pca_pre_model = PCA(n_components=min(pca_pre, X_scaled.shape[1]), random_state=42)
    X_pre = pca_pre_model.fit_transform(X_scaled)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=n_iter)
    X_tsne = tsne.fit_transform(X_pre)

fig_tsne, ax = plt.subplots(figsize=(9, 5.5))
fig_tsne.patch.set_facecolor('#f5f5f0')
ax.set_facecolor('white')
sc2 = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.7, s=25)
plt.colorbar(sc2, ax=ax, label='Class Label')
ax.set_xlabel('t-SNE Component 1', color='#555')
ax.set_ylabel('t-SNE Component 2', color='#555')
ax.set_title(f't-SNE — 2D Projection  (perplexity={perplexity})', color='#1a1a2e', fontsize=13)
ax.spines[['top','right']].set_visible(False)
ax.spines[['left','bottom']].set_color('#ddd')
ax.tick_params(colors='#aaa')
ax.grid(True, color='#eee', linestyle='--')
st.pyplot(fig_tsne)

st.divider()

# ── COMPARISON ────────────────────────────────────────────────────────────────
st.markdown('<div class="tab-header">③ PCA vs t-SNE — Side by Side</div>', unsafe_allow_html=True)

fig_cmp, axes = plt.subplots(1, 2, figsize=(15, 5.5))
fig_cmp.patch.set_facecolor('#f5f5f0')

for ax in axes:
    ax.set_facecolor('white')
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#ddd')
    ax.tick_params(colors='#aaa')
    ax.grid(True, color='#eee', linestyle='--')

sc1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.65, s=20)
axes[0].set_title('PCA  (linear)', color='#1a1a2e', fontsize=12)
axes[0].set_xlabel('PC1', color='#555')
axes[0].set_ylabel('PC2', color='#555')
plt.colorbar(sc1, ax=axes[0])

sc2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.65, s=20)
axes[1].set_title('t-SNE  (non-linear)', color='#1a1a2e', fontsize=12)
axes[1].set_xlabel('Component 1', color='#555')
axes[1].set_ylabel('Component 2', color='#555')
plt.colorbar(sc2, ax=axes[1])

plt.suptitle('Dimensionality Reduction Comparison — Digits Dataset', fontsize=13, color='#1a1a2e', y=1.01)
plt.tight_layout()
st.pyplot(fig_cmp)

st.markdown("""
| | PCA | t-SNE |
|---|---|---|
| Type | Linear | Non-linear |
| Speed | Fast | Slow |
| Best for | Preprocessing, compression | Visualization only |
| Interpretable | Yes (loadings) | No |
| New data | Yes (transform) | No |
""")

st.divider()

# download
df_out = pd.DataFrame({'pca_1': X_pca[:,0], 'pca_2': X_pca[:,1],
                        'tsne_1': X_tsne[:,0], 'tsne_2': X_tsne[:,1], 'label': y})
csv = df_out.to_csv(index=False).encode()
st.download_button("⬇️ Download Reduced Data", csv, "digits_reduced.csv", "text/csv")
