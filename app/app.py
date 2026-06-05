import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Unsupervised Learning Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Space Mono', monospace; }

    .main { background-color: #0e1117; }

    .metric-card {
        background: linear-gradient(135deg, #1e2130 0%, #252840 100%);
        border: 1px solid #3a3f6b;
        border-radius: 12px;
        padding: 18px 22px;
        text-align: center;
        margin-bottom: 12px;
    }
    .metric-card .label { font-size: 0.78rem; color: #8b90c8; letter-spacing: 0.08em; text-transform: uppercase; }
    .metric-card .value { font-size: 1.7rem; font-weight: 700; color: #a5b4fc; font-family: 'Space Mono', monospace; }

    .algo-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    .badge-dbscan { background: #1a3a4a; color: #38bdf8; border: 1px solid #0ea5e9; }
    .badge-gmm    { background: #2a1a4a; color: #c084fc; border: 1px solid #a855f7; }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #1e2130; border-radius: 8px;
        padding: 8px 20px; color: #8b90c8;
        font-family: 'Space Mono', monospace; font-size: 0.82rem;
    }
    .stTabs [aria-selected="true"] { background: #3a3f6b !important; color: #e0e7ff !important; }

    section[data-testid="stSidebar"] { background: #131722 !important; }
</style>
""", unsafe_allow_html=True)


# ── Helper: load data ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "data", "clustering_dataset.csv")
    return pd.read_csv(path)

# ── Helper: scale ─────────────────────────────────────────────────────────────
def scale(df):
    scaler = StandardScaler()
    return scaler.fit_transform(df), scaler

# ── Helper: metrics ───────────────────────────────────────────────────────────
def compute_metrics(X, labels):
    mask = labels != -1
    n_clusters = len(set(labels[mask])) if mask.sum() > 0 else 0
    n_noise = (labels == -1).sum()
    metrics = {"n_clusters": n_clusters, "n_noise": n_noise}
    if n_clusters >= 2 and mask.sum() > n_clusters:
        metrics["silhouette"]  = round(silhouette_score(X[mask], labels[mask]), 4)
        metrics["davies_bouldin"] = round(davies_bouldin_score(X[mask], labels[mask]), 4)
        metrics["calinski"]    = round(calinski_harabasz_score(X[mask], labels[mask]), 4)
    else:
        metrics["silhouette"] = metrics["davies_bouldin"] = metrics["calinski"] = "N/A"
    return metrics

# ── Palette ───────────────────────────────────────────────────────────────────
PALETTE = [
    "#38bdf8","#a78bfa","#34d399","#fb923c","#f472b6",
    "#facc15","#60a5fa","#e879f9","#4ade80","#f87171",
]

def cluster_colors(labels):
    unique = sorted(set(labels))
    cmap = {}
    c_idx = 0
    for u in unique:
        cmap[u] = "#ef4444" if u == -1 else PALETTE[c_idx % len(PALETTE)]
        if u != -1: c_idx += 1
    return np.array([cmap[l] for l in labels])


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔬 Algorithm")
    algorithm = st.radio("Select model", ["DBSCAN", "GMM (Gaussian Mixture)"], label_visibility="collapsed")

    st.markdown("---")

    df_raw = load_data()
    cols = df_raw.columns.tolist()
    feat_x = st.selectbox("Feature X", cols, index=0)
    feat_y = st.selectbox("Feature Y", cols, index=1 if len(cols) > 1 else 0)

    st.markdown("---")

    if "DBSCAN" in algorithm:
        st.markdown('<span class="algo-badge badge-dbscan">DBSCAN Hyperparameters</span>', unsafe_allow_html=True)
        eps       = st.slider("ε (eps) — neighborhood radius", 0.05, 3.0, 0.5, 0.05,
                              help="Max distance between two samples to be in the same neighborhood.")
        min_samp  = st.slider("min_samples — core point threshold", 2, 30, 5, 1,
                              help="Minimum points in ε-neighborhood to form a core point.")
        metric    = st.selectbox("Distance metric", ["euclidean","manhattan","chebyshev"])
        algorithm_dbscan = st.selectbox("Algorithm", ["auto","ball_tree","kd_tree","brute"])
        leaf_size = st.slider("Leaf size (ball_tree / kd_tree)", 10, 100, 30, 5)

    else:
        st.markdown('<span class="algo-badge badge-gmm">GMM Hyperparameters</span>', unsafe_allow_html=True)
        n_components  = st.slider("n_components — number of Gaussians", 2, 15, 5, 1,
                                  help="Number of mixture components (clusters).")
        covariance_type = st.selectbox("Covariance type",
                                       ["full","tied","diag","spherical"],
                                       help="Shape of covariance matrices.")
        max_iter      = st.slider("max_iter", 50, 500, 100, 50)
        n_init        = st.slider("n_init — number of initializations", 1, 10, 1, 1)
        init_params   = st.selectbox("init_params", ["kmeans","random","k-means++","random_from_data"])
        reg_covar     = st.select_slider("reg_covar (regularization)",
                                         options=[1e-6,1e-5,1e-4,1e-3,0.01,0.1],
                                         value=1e-6)
        tol           = st.select_slider("tol (convergence threshold)",
                                         options=[1e-5,1e-4,1e-3,0.01,0.1],
                                         value=1e-3)

    st.markdown("---")
    scale_data = st.checkbox("Standardise features", value=True)
    run_btn    = st.button("▶  Run", use_container_width=True, type="primary")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🔬 Unsupervised Learning Explorer")
st.markdown("**Dataset:** `clustering_dataset.csv` — 530 samples, 2 features (5 blobs + noise)   |   "
            "**Models:** DBSCAN · Gaussian Mixture Model")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_cluster, tab_eval, tab_data, tab_theory = st.tabs(
    ["📊 Clustering", "📈 Evaluation", "🗃 Data Explorer", "📖 Theory"])

X_raw = df_raw[[feat_x, feat_y]].values
X_scaled, scaler = scale(df_raw[[feat_x, feat_y]])
X = X_scaled if scale_data else X_raw

# ── Run model ─────────────────────────────────────────────────────────────────
labels = None
model  = None

if run_btn or "labels" not in st.session_state:
    if "DBSCAN" in algorithm:
        model  = DBSCAN(eps=eps, min_samples=min_samp,
                        metric=metric, algorithm=algorithm_dbscan,
                        leaf_size=leaf_size)
        labels = model.fit_predict(X)
    else:
        model  = GaussianMixture(n_components=n_components,
                                 covariance_type=covariance_type,
                                 max_iter=max_iter, n_init=n_init,
                                 init_params=init_params,
                                 reg_covar=reg_covar, tol=tol,
                                 random_state=42)
        model.fit(X)
        labels = model.predict(X)
    st.session_state["labels"] = labels
    st.session_state["model"]  = model
else:
    labels = st.session_state["labels"]
    model  = st.session_state["model"]

colors = cluster_colors(labels)
metrics = compute_metrics(X, labels)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Clustering
# ═══════════════════════════════════════════════════════════════════════════════
with tab_cluster:
    # Metric cards
    c1, c2, c3, c4 = st.columns(4)
    for col, lbl, val in zip(
        [c1, c2, c3, c4],
        ["Clusters Found", "Noise Points", "Silhouette ↑", "Davies-Bouldin ↓"],
        [metrics["n_clusters"], metrics["n_noise"],
         metrics["silhouette"], metrics["davies_bouldin"]]
    ):
        col.markdown(
            f'<div class="metric-card"><div class="label">{lbl}</div><div class="value">{val}</div></div>',
            unsafe_allow_html=True)

    st.markdown("### Cluster Scatter Plot")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0e1117")

    for ax, (X_plot, title) in zip(axes, [
        (X_raw, "Original Feature Space"),
        (X,     "Scaled Feature Space" if scale_data else "Original Feature Space")
    ]):
        ax.set_facecolor("#13161f")
        ax.scatter(X_plot[:, 0], X_plot[:, 1],
                   c=colors, s=25, alpha=0.85, linewidths=0)
        # GMM ellipses
        if "GMM" in algorithm and model is not None:
            from matplotlib.patches import Ellipse
            import matplotlib.transforms as mpl_transforms
            for i, (mean, cov) in enumerate(zip(model.means_, model.covariances_)):
                if covariance_type == "full":
                    C = cov
                elif covariance_type == "tied":
                    C = model.covariances_
                elif covariance_type == "diag":
                    C = np.diag(cov)
                else:
                    C = np.eye(2) * cov
                vals, vecs = np.linalg.eigh(C)
                order = vals.argsort()[::-1]
                vals, vecs = vals[order], vecs[:, order]
                theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                w, h = 2 * np.sqrt(vals)
                # project mean back if scaled
                if scale_data and X_plot is X_raw:
                    m = scaler.inverse_transform(mean.reshape(1,-1))[0]
                    w *= scaler.scale_[0]; h *= scaler.scale_[1]
                else:
                    m = mean
                ell = Ellipse(xy=m, width=w, height=h, angle=theta,
                              edgecolor=PALETTE[i % len(PALETTE)],
                              facecolor="none", linewidth=1.5, linestyle="--", alpha=0.7)
                ax.add_patch(ell)

        ax.set_title(title, color="#e0e7ff", fontsize=11, pad=10)
        ax.set_xlabel(feat_x, color="#8b90c8", fontsize=9)
        ax.set_ylabel(feat_y, color="#8b90c8", fontsize=9)
        ax.tick_params(colors="#8b90c8")
        for spine in ax.spines.values():
            spine.set_edgecolor("#3a3f6b")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Cluster size bar chart
    st.markdown("### Cluster Size Distribution")
    unique_labels, counts = np.unique(labels, return_counts=True)
    fig2, ax2 = plt.subplots(figsize=(10, 3), facecolor="#0e1117")
    ax2.set_facecolor("#13161f")
    bar_colors = [cluster_colors(np.array([l]))[0] for l in unique_labels]
    bar_labels = [f"Noise" if l == -1 else f"Cluster {l}" for l in unique_labels]
    bars = ax2.bar(bar_labels, counts, color=bar_colors, edgecolor="#0e1117", linewidth=0.8)
    for bar, cnt in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 str(cnt), ha="center", va="bottom", color="#e0e7ff", fontsize=9)
    ax2.set_ylabel("Count", color="#8b90c8")
    ax2.tick_params(colors="#8b90c8", axis="x", rotation=30)
    ax2.tick_params(colors="#8b90c8", axis="y")
    for spine in ax2.spines.values(): spine.set_edgecolor("#3a3f6b")
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close()

    # GMM probability contour
    if "GMM" in algorithm and model is not None:
        st.markdown("### GMM Probability Density Contour")
        x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
        y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        Z = -model.score_samples(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        fig3, ax3 = plt.subplots(figsize=(8, 5), facecolor="#0e1117")
        ax3.set_facecolor("#13161f")
        cf = ax3.contourf(xx, yy, Z, levels=30, cmap="plasma", alpha=0.7)
        ax3.scatter(X[:, 0], X[:, 1], c=colors, s=15, alpha=0.6, linewidths=0)
        plt.colorbar(cf, ax=ax3, label="−log likelihood")
        ax3.set_title("GMM Density Landscape", color="#e0e7ff")
        ax3.tick_params(colors="#8b90c8")
        for spine in ax3.spines.values(): spine.set_edgecolor("#3a3f6b")
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Evaluation
# ═══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown("### 📊 Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    for col, lbl, val, note in zip(
        [col1, col2, col3],
        ["Silhouette Score", "Davies-Bouldin Index", "Calinski-Harabasz"],
        [metrics["silhouette"], metrics["davies_bouldin"], metrics["calinski"]],
        ["Range [−1, 1]. Higher = better separated clusters.",
         "Lower = better. Measures intra/inter-cluster similarity.",
         "Higher = denser, well-separated clusters."]
    ):
        col.markdown(
            f'<div class="metric-card"><div class="label">{lbl}</div>'
            f'<div class="value">{val}</div></div>',
            unsafe_allow_html=True)
        col.caption(note)

    # Parameter sweep
    st.markdown("---")
    if "DBSCAN" in algorithm:
        st.markdown("### 🔍 DBSCAN — eps vs min_samples Heatmap (Silhouette)")
        eps_vals   = np.arange(0.1, 2.1, 0.2)
        ms_vals    = range(2, 12, 2)
        sil_matrix = np.zeros((len(list(ms_vals)), len(eps_vals)))
        for i, ms in enumerate(ms_vals):
            for j, ep in enumerate(eps_vals):
                l = DBSCAN(eps=ep, min_samples=ms, metric=metric).fit_predict(X)
                mask = l != -1
                n_cl = len(set(l[mask]))
                if n_cl >= 2 and mask.sum() > n_cl:
                    sil_matrix[i, j] = silhouette_score(X[mask], l[mask])
        fig_h, ax_h = plt.subplots(figsize=(9, 4), facecolor="#0e1117")
        ax_h.set_facecolor("#13161f")
        im = ax_h.imshow(sil_matrix, aspect="auto", cmap="magma",
                         vmin=0, vmax=max(sil_matrix.max(), 0.01))
        ax_h.set_xticks(range(len(eps_vals)))
        ax_h.set_xticklabels([f"{e:.1f}" for e in eps_vals], color="#8b90c8", fontsize=8)
        ax_h.set_yticks(range(len(list(ms_vals))))
        ax_h.set_yticklabels(list(ms_vals), color="#8b90c8")
        ax_h.set_xlabel("eps", color="#8b90c8")
        ax_h.set_ylabel("min_samples", color="#8b90c8")
        ax_h.set_title("Silhouette Score Sweep", color="#e0e7ff")
        plt.colorbar(im, ax=ax_h)
        for i in range(sil_matrix.shape[0]):
            for j in range(sil_matrix.shape[1]):
                ax_h.text(j, i, f"{sil_matrix[i,j]:.2f}",
                          ha="center", va="center", color="white", fontsize=7)
        plt.tight_layout()
        st.pyplot(fig_h, use_container_width=True)
        plt.close()

    else:
        st.markdown("### 🔍 GMM — BIC / AIC Component Selection")
        comp_range = range(2, 13)
        bics, aics = [], []
        for nc in comp_range:
            gm = GaussianMixture(n_components=nc, covariance_type=covariance_type,
                                 max_iter=max_iter, random_state=42)
            gm.fit(X)
            bics.append(gm.bic(X))
            aics.append(gm.aic(X))

        fig_ba, ax_ba = plt.subplots(figsize=(9, 4), facecolor="#0e1117")
        ax_ba.set_facecolor("#13161f")
        ax_ba.plot(list(comp_range), bics, "o-", color="#38bdf8", label="BIC", linewidth=2)
        ax_ba.plot(list(comp_range), aics, "s-", color="#a78bfa", label="AIC", linewidth=2)
        ax_ba.axvline(n_components, color="#f472b6", linestyle="--", alpha=0.8, label=f"Current n={n_components}")
        ax_ba.set_xlabel("n_components", color="#8b90c8")
        ax_ba.set_ylabel("Score (lower = better)", color="#8b90c8")
        ax_ba.set_title("BIC / AIC vs Number of Components", color="#e0e7ff")
        ax_ba.tick_params(colors="#8b90c8")
        ax_ba.legend(facecolor="#1e2130", labelcolor="#e0e7ff")
        for spine in ax_ba.spines.values(): spine.set_edgecolor("#3a3f6b")
        plt.tight_layout()
        st.pyplot(fig_ba, use_container_width=True)
        plt.close()

        st.markdown("### GMM Convergence")
        fig_conv, ax_conv = plt.subplots(figsize=(7, 3), facecolor="#0e1117")
        ax_conv.set_facecolor("#13161f")
        ax_conv.plot(range(1, len(model.lower_bound_) + 1 if hasattr(model, "lower_bound_") and hasattr(model.lower_bound_, "__len__") else 2),
                     [model.lower_bound_] if not hasattr(model.lower_bound_, "__len__") else model.lower_bound_,
                     "o-", color="#34d399")
        ax_conv.set_xlabel("Iteration", color="#8b90c8")
        ax_conv.set_ylabel("Lower Bound", color="#8b90c8")
        ax_conv.set_title("EM Lower Bound (Convergence)", color="#e0e7ff")
        ax_conv.tick_params(colors="#8b90c8")
        for spine in ax_conv.spines.values(): spine.set_edgecolor("#3a3f6b")
        plt.tight_layout()
        st.pyplot(fig_conv, use_container_width=True)
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Data Explorer
# ═══════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.markdown("### 🗃 Raw Dataset")
    df_view = df_raw.copy()
    df_view["cluster_label"] = labels
    st.dataframe(df_view.style.background_gradient(cmap="plasma", subset=[feat_x, feat_y]),
                 use_container_width=True, height=300)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Feature Distributions")
        fig_d, axes_d = plt.subplots(1, 2, figsize=(8, 3), facecolor="#0e1117")
        for ax, col_ in zip(axes_d, [feat_x, feat_y]):
            ax.set_facecolor("#13161f")
            ax.hist(df_raw[col_], bins=30, color="#38bdf8", edgecolor="#0e1117", alpha=0.85)
            ax.set_title(col_, color="#e0e7ff", fontsize=10)
            ax.tick_params(colors="#8b90c8")
            for spine in ax.spines.values(): spine.set_edgecolor("#3a3f6b")
        plt.tight_layout()
        st.pyplot(fig_d, use_container_width=True)
        plt.close()

    with col_b:
        st.markdown("#### Correlation Heatmap")
        fig_c, ax_c = plt.subplots(figsize=(4, 3), facecolor="#0e1117")
        ax_c.set_facecolor("#13161f")
        sns.heatmap(df_raw.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                    ax=ax_c, cbar_kws={"shrink": 0.8})
        ax_c.tick_params(colors="#8b90c8")
        ax_c.set_title("Feature Correlation", color="#e0e7ff")
        plt.tight_layout()
        st.pyplot(fig_c, use_container_width=True)
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Theory
# ═══════════════════════════════════════════════════════════════════════════════
with tab_theory:
    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown('<span class="algo-badge badge-dbscan">DBSCAN</span>', unsafe_allow_html=True)
        st.markdown("""
**Density-Based Spatial Clustering of Applications with Noise**

DBSCAN groups together points that are closely packed (many neighbors within radius **ε**) 
and marks outliers in low-density regions as **noise**.

| Hyperparameter | Role |
|---|---|
| `eps` (ε) | Neighborhood radius. Smaller → finer clusters, more noise |
| `min_samples` | Min points to form core point. Higher → fewer, denser clusters |
| `metric` | Distance function (euclidean, manhattan…) |
| `algorithm` | Nearest-neighbor search method |
| `leaf_size` | Affects tree speed/memory trade-off |

**Point types:**  
🔵 **Core** — ≥ min_samples within ε  
🟡 **Border** — within ε of a core, but not core itself  
🔴 **Noise** — not within ε of any core point  

**Strengths:** Finds arbitrary-shaped clusters, robust to outliers, no need to specify K  
**Weaknesses:** Struggles with varying-density clusters, sensitive to ε
        """)

    with col_t2:
        st.markdown('<span class="algo-badge badge-gmm">GMM</span>', unsafe_allow_html=True)
        st.markdown("""
**Gaussian Mixture Model**

GMM assumes data is generated from a mixture of **K Gaussian distributions**. 
Uses the **EM algorithm** (Expectation–Maximisation) to find parameters.

| Hyperparameter | Role |
|---|---|
| `n_components` | Number of Gaussians (clusters) |
| `covariance_type` | Shape: `full`, `tied`, `diag`, `spherical` |
| `max_iter` | EM iteration limit |
| `n_init` | Number of random restarts |
| `init_params` | Initialisation strategy |
| `reg_covar` | Regularisation for numerical stability |
| `tol` | Convergence threshold |

**Covariance types:**  
`full` — each cluster has its own covariance matrix  
`tied` — all clusters share one matrix  
`diag` — diagonal matrices only  
`spherical` — one variance per cluster  

**Strengths:** Soft assignments (probabilities), flexible elliptical shapes, BIC/AIC for model selection  
**Weaknesses:** Assumes Gaussian distributions, sensitive to initialisation
        """)