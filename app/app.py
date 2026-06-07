import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DR Lab · PCA & t-SNE",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp {
    background: #080c14;
    color: #e2e8f0;
}

[data-testid="stSidebar"] {
    background: #0b0f1c;
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

.hero-banner {
    background: linear-gradient(135deg, #0f1f3d 0%, #0c1a35 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 1rem;
}
.hero-left .hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0 0 0.3rem 0;
    color: #f1f5f9;
    letter-spacing: -0.5px;
}
.hero-left .hero-sub { color: #64748b; font-size: 0.95rem; margin: 0; }
.hero-badges { display: flex; gap: 0.5rem; flex-wrap: wrap; }
.badge {
    padding: 0.35rem 0.9rem;
    border-radius: 100px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.3px;
}
.badge-pca  { background: rgba(99,179,237,0.12); color: #63b3ed; border: 1px solid rgba(99,179,237,0.3); }
.badge-tsne { background: rgba(167,139,250,0.12); color: #a78bfa; border: 1px solid rgba(167,139,250,0.3); }
.badge-data { background: rgba(52,211,153,0.1);  color: #34d399; border: 1px solid rgba(52,211,153,0.25); }

.method-tabs { display: flex; gap: 0; margin-bottom: 1.5rem; border-radius: 10px; overflow: hidden; border: 1px solid rgba(255,255,255,0.08); width: fit-content; }
.method-tab {
    padding: 0.6rem 1.8rem;
    font-size: 0.9rem;
    font-weight: 600;
    cursor: pointer;
    background: rgba(255,255,255,0.03);
    color: #64748b;
    transition: all 0.2s;
}
.method-tab-pca-active  { background: rgba(99,179,237,0.15); color: #63b3ed; }
.method-tab-tsne-active { background: rgba(167,139,250,0.15); color: #a78bfa; }

.metric-row { display: flex; gap: 0.75rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.metric-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1rem 1.3rem;
    flex: 1; min-width: 120px;
    transition: all 0.2s;
}
.metric-card:hover { border-color: rgba(255,255,255,0.15); background: rgba(255,255,255,0.05); }
.metric-label { font-size: 0.68rem; color: #475569; text-transform: uppercase; letter-spacing: 1.2px; font-weight: 600; margin-bottom: 0.3rem; }
.metric-value { font-size: 1.7rem; font-weight: 700; font-family: 'DM Mono', monospace; }
.metric-pca  .metric-value { color: #63b3ed; }
.metric-tsne .metric-value { color: #a78bfa; }
.metric-neutral .metric-value { color: #34d399; }
.metric-unit { font-size: 0.78rem; color: #475569; }

.section-header {
    font-size: 1rem; font-weight: 600; color: #94a3b8;
    border-left: 3px solid rgba(255,255,255,0.2);
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem 0;
    letter-spacing: 0.2px;
}
.section-header-pca  { border-left-color: #63b3ed; color: #90cdf4; }
.section-header-tsne { border-left-color: #a78bfa; color: #c4b5fd; }
.section-header-both { border-left-color: #34d399; color: #6ee7b7; }

.stButton > button {
    background: linear-gradient(135deg, #1e3a5f, #162d4a);
    color: white !important;
    border: 1px solid rgba(99,179,237,0.3) !important;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
    transition: all 0.2s;
    width: 100%;
}
.stButton > button:hover { background: linear-gradient(135deg, #2a5080, #1e3a5f) !important; transform: translateY(-1px); }

[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
}

.comparison-panel {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-top: 1rem;
}
.comparison-title { font-size: 0.85rem; font-weight: 600; color: #64748b; margin-bottom: 0.8rem; letter-spacing: 0.5px; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

# ─── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-left">
    <div class="hero-title">🧬 Dimensionality Reduction Lab</div>
    <div class="hero-sub">Principal Component Analysis &amp; t-SNE · Side-by-Side Comparison</div>
  </div>
  <div class="hero-badges">
    <span class="badge badge-pca">PCA · Linear</span>
    <span class="badge badge-tsne">t-SNE · Non-linear</span>
    <span class="badge badge-data">Digits Dataset</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'digits.csv')
df_raw = pd.read_csv(DATA_PATH)
feature_cols = [c for c in df_raw.columns if c != 'target']

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")
    st.success(f"Digits Dataset · {df_raw.shape[0]} samples · {len(feature_cols)} features · 10 classes")

    scaling = st.selectbox("Feature Scaling", ["StandardScaler", "None"])
    viz_dim = st.radio("Visualization", ["2D", "3D"], horizontal=True)
    random_state = st.number_input("Random State", value=42, step=1)

    st.markdown("---")
    st.markdown("**🔷 PCA Hyperparameters**")
    pca_n = st.slider("PCA Components", 2, min(20, len(feature_cols)), 10)
    pca_whiten = st.checkbox("Whiten", value=False)
    pca_solver = st.selectbox("SVD Solver", ["auto", "full", "randomized"])

    st.markdown("---")
    st.markdown("**🌀 t-SNE Hyperparameters**")
    tsne_perp = st.slider("Perplexity", 5, 80, 30, step=5)
    tsne_iter = st.slider("Iterations", 250, 1500, 1000, step=250)
    tsne_lr = st.select_slider("Learning Rate", [10, 50, 100, 200, 500], value=200)
    tsne_ee = st.slider("Early Exaggeration", 4.0, 20.0, 12.0, step=2.0)
    tsne_init = st.selectbox("Initialization", ["pca", "random"])
    tsne_metric = st.selectbox("Distance Metric", ["euclidean", "cosine"])

    st.markdown("---")
    run_btn = st.button("▶ Run Both Methods", use_container_width=True)

# ─── Compute ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_all(pca_n, pca_whiten, pca_solver, tsne_perp, tsne_iter,
                tsne_lr, tsne_ee, tsne_init, tsne_metric, scaling, random_state, viz_dim):
    df = pd.read_csv(DATA_PATH)
    X = df[[c for c in df.columns if c != 'target']].values
    y = df['target'].values

    if scaling == "StandardScaler":
        X_s = StandardScaler().fit_transform(X)
    else:
        X_s = X.copy()

    # PCA
    pca = PCA(n_components=pca_n, whiten=pca_whiten, svd_solver=pca_solver, random_state=random_state)
    X_pca = pca.fit_transform(X_s)

    # t-SNE (on PCA-reduced for speed)
    n_dim = 3 if viz_dim == "3D" else 2
    X_pre = PCA(n_components=min(50, X_s.shape[1]), random_state=random_state).fit_transform(X_s)
    tsne = TSNE(n_components=n_dim, perplexity=tsne_perp, n_iter=tsne_iter, learning_rate=tsne_lr,
                early_exaggeration=tsne_ee, init=tsne_init, metric=tsne_metric, random_state=random_state, n_jobs=-1)
    X_tsne = tsne.fit_transform(X_pre)

    return pca, X_pca, X_tsne, y, tsne.kl_divergence_

if run_btn or True:
    with st.spinner("Running PCA & t-SNE..."):
        pca, X_pca, X_tsne, y, kl_div = compute_all(
            pca_n, pca_whiten, pca_solver, tsne_perp, tsne_iter,
            tsne_lr, tsne_ee, tsne_init, tsne_metric, scaling, int(random_state), viz_dim
        )

    ev = pca.explained_variance_ratio_
    y_str = [str(int(v)) for v in y]

    PLOTLY_THEME = dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,12,20,0.9)",
        font=dict(family="DM Sans", color="#94a3b8"),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    COLOR_SEQ = px.colors.qualitative.Alphabet[:10]

    # ── Metric Cards ───────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card metric-pca">
        <div class="metric-label">PCA Components</div>
        <div class="metric-value">{pca_n}</div>
      </div>
      <div class="metric-card metric-pca">
        <div class="metric-label">Variance Explained</div>
        <div class="metric-value">{ev.sum()*100:.1f}<span class="metric-unit">%</span></div>
      </div>
      <div class="metric-card metric-tsne">
        <div class="metric-label">t-SNE Perplexity</div>
        <div class="metric-value">{tsne_perp}</div>
      </div>
      <div class="metric-card metric-tsne">
        <div class="metric-label">KL Divergence</div>
        <div class="metric-value">{kl_div:.3f}</div>
      </div>
      <div class="metric-card metric-neutral">
        <div class="metric-label">Samples</div>
        <div class="metric-value">{len(y)}</div>
      </div>
      <div class="metric-card metric-neutral">
        <div class="metric-label">Digit Classes</div>
        <div class="metric-value">10</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Side by Side Projections ───────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f'<div class="section-header section-header-pca">🔷 PCA Projection — {viz_dim}</div>', unsafe_allow_html=True)
        if viz_dim == "2D":
            fig_pca = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=y_str,
                                  color_discrete_sequence=COLOR_SEQ,
                                  labels={"x": "PC1", "y": "PC2", "color": "Digit"}, opacity=0.75)
            fig_pca.update_traces(marker=dict(size=5, line=dict(width=0.3, color="rgba(0,0,0,0.3)")))
        else:
            fig_pca = px.scatter_3d(x=X_pca[:, 0], y=X_pca[:, 1],
                                     z=X_pca[:, 2] if pca_n >= 3 else np.zeros(len(y)),
                                     color=y_str, color_discrete_sequence=COLOR_SEQ,
                                     labels={"x": "PC1", "y": "PC2", "z": "PC3", "color": "Digit"}, opacity=0.75)
            fig_pca.update_traces(marker=dict(size=3))
        fig_pca.update_layout(**PLOTLY_THEME, height=420, title=dict(text=f"PC1 vs PC2 · {ev[0]*100:.1f}% + {ev[1]*100:.1f}% = {(ev[0]+ev[1])*100:.1f}%", font=dict(size=12, color="#90cdf4")))
        st.plotly_chart(fig_pca, use_container_width=True)

    with col2:
        st.markdown(f'<div class="section-header section-header-tsne">🌀 t-SNE Projection — {viz_dim}</div>', unsafe_allow_html=True)
        if viz_dim == "2D":
            fig_tsne = px.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], color=y_str,
                                   color_discrete_sequence=COLOR_SEQ,
                                   labels={"x": "Dim 1", "y": "Dim 2", "color": "Digit"}, opacity=0.75)
            fig_tsne.update_traces(marker=dict(size=5, line=dict(width=0.3, color="rgba(0,0,0,0.3)")))
        else:
            fig_tsne = px.scatter_3d(x=X_tsne[:, 0], y=X_tsne[:, 1], z=X_tsne[:, 2],
                                      color=y_str, color_discrete_sequence=COLOR_SEQ,
                                      labels={"x": "Dim1", "y": "Dim2", "z": "Dim3", "color": "Digit"}, opacity=0.75)
            fig_tsne.update_traces(marker=dict(size=3))
        fig_tsne.update_layout(**PLOTLY_THEME, height=420, title=dict(text=f"KL divergence: {kl_div:.4f}", font=dict(size=12, color="#c4b5fd")))
        st.plotly_chart(fig_tsne, use_container_width=True)

    # ── PCA Scree Plot ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header section-header-pca">📊 PCA — Explained Variance</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        comp_labels = [f"PC{i+1}" for i in range(pca_n)]
        cumulative = np.cumsum(ev) * 100
        fig_scree = go.Figure()
        fig_scree.add_bar(x=comp_labels, y=ev * 100, name="Individual",
                          marker_color="#63b3ed", opacity=0.8)
        fig_scree.add_scatter(x=comp_labels, y=cumulative, name="Cumulative",
                              mode="lines+markers", line=dict(color="#f6ad55", width=2),
                              marker=dict(size=6), yaxis="y2")
        fig_scree.update_layout(**PLOTLY_THEME, height=320,
                                 yaxis=dict(title="Variance (%)", gridcolor="rgba(99,179,237,0.08)"),
                                 yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0,105]),
                                 legend=dict(orientation="h", y=1.1),
                                 title=dict(text="Scree Plot", font=dict(size=12, color="#90cdf4")))
        st.plotly_chart(fig_scree, use_container_width=True)

    with col4:
        # Feature contribution (top features to PC1)
        pc1_abs = np.abs(pca.components_[0])
        top10_idx = np.argsort(pc1_abs)[::-1][:10]
        fig_feat = go.Figure(go.Bar(
            x=pc1_abs[top10_idx],
            y=[feature_cols[i] for i in top10_idx],
            orientation='h',
            marker=dict(color=pc1_abs[top10_idx], colorscale="Blues_r", showscale=False)
        ))
        fig_feat.update_layout(**PLOTLY_THEME, height=320,
                                xaxis_title="|Loading|",
                                yaxis=dict(autorange="reversed"),
                                title=dict(text="Top 10 Features → PC1", font=dict(size=12, color="#90cdf4")))
        st.plotly_chart(fig_feat, use_container_width=True)

    # ── Comparison Table ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header section-header-both">⚖️ Method Comparison</div>', unsafe_allow_html=True)
    comp_df = pd.DataFrame({
        "Property": ["Type", "Preserves Global Structure", "Preserves Local Structure",
                     "Deterministic", "Scalability", "Interpretability", "Best Use Case"],
        "🔷 PCA": ["Linear", "✅ Yes", "⚠️ Partial", "✅ Yes", "✅ High",
                   "✅ High (loadings)", "Variance analysis, preprocessing"],
        "🌀 t-SNE": ["Non-linear", "⚠️ Partial", "✅ Yes", "❌ No (stochastic)",
                     "⚠️ Medium", "❌ Low (no loadings)", "Cluster visualization, exploration"],
    })
    st.dataframe(comp_df, hide_index=True, use_container_width=True)

    # ── Loadings Heatmap ────────────────────────────────────────────────────────
    with st.expander("🧩 PCA Loadings Heatmap (first 5 PCs × 20 features)"):
        n_feat_show = min(20, len(feature_cols))
        top_feat_idx = np.argsort(np.abs(pca.components_[:5]).sum(axis=0))[::-1][:n_feat_show]
        loadings_df = pd.DataFrame(
            pca.components_[:5, :][:, top_feat_idx],
            columns=[feature_cols[i] for i in top_feat_idx],
            index=[f"PC{i+1}" for i in range(5)]
        )
        fig_heat = px.imshow(loadings_df, color_continuous_scale="RdBu_r",
                              aspect="auto", zmin=-1, zmax=1, text_auto=".2f")
        fig_heat.update_layout(**PLOTLY_THEME, height=250, margin=dict(l=20, r=20, t=30, b=40),
                                xaxis=dict(tickangle=-40, tickfont=dict(size=9)))
        st.plotly_chart(fig_heat, use_container_width=True)

    with st.expander("📋 Transformed Data Preview (first 30 rows)"):
        tab1, tab2 = st.tabs(["PCA", "t-SNE"])
        with tab1:
            pca_df = pd.DataFrame(X_pca[:30], columns=[f"PC{i+1}" for i in range(pca_n)])
            pca_df["Digit"] = y[:30].astype(int)
            st.dataframe(pca_df, use_container_width=True)
        with tab2:
            tsne_df = pd.DataFrame(X_tsne[:30], columns=[f"Dim{i+1}" for i in range(X_tsne.shape[1])])
            tsne_df["Digit"] = y[:30].astype(int)
            st.dataframe(tsne_df, use_container_width=True)
