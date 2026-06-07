import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DBSCAN Clustering",
    page_icon="🔵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Fira+Code:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }

.stApp {
    background: radial-gradient(ellipse at top left, #071020 0%, #040c18 40%, #060e1c 100%);
    color: #dce8f5;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060f1e 0%, #07111f 100%);
    border-right: 1px solid rgba(56,189,248,0.12);
}
[data-testid="stSidebar"] * { color: #bae6fd !important; }
[data-testid="stSidebar"] .stMarkdown h3 { color: #38bdf8 !important; font-size: 1rem !important; }

/* Hero */
.hero-banner {
    background: linear-gradient(135deg, #0c1e38 0%, #091830 60%, #0b1c34 100%);
    border: 1px solid rgba(56,189,248,0.18);
    border-radius: 18px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::after {
    content: 'DBSCAN';
    position: absolute;
    right: 2rem; top: 50%;
    transform: translateY(-50%);
    font-size: 5rem; font-weight: 800;
    color: rgba(56,189,248,0.04);
    letter-spacing: -2px;
    pointer-events: none;
    font-family: 'Outfit', sans-serif;
}
.hero-title {
    font-size: 2.3rem; font-weight: 800;
    background: linear-gradient(90deg, #38bdf8, #7dd3fc, #bae6fd);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 0.35rem 0; letter-spacing: -0.5px;
}
.hero-sub { color: #64748b; font-size: 0.95rem; margin: 0; }
.hero-tags { display: flex; gap: 0.5rem; margin-top: 0.8rem; flex-wrap: wrap; }
.tag {
    padding: 0.25rem 0.75rem; border-radius: 100px;
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.3px;
}
.tag-algo { background: rgba(56,189,248,0.1); color: #38bdf8; border: 1px solid rgba(56,189,248,0.25); }
.tag-data { background: rgba(52,211,153,0.08); color: #34d399; border: 1px solid rgba(52,211,153,0.2); }
.tag-type { background: rgba(251,191,36,0.08); color: #fbbf24; border: 1px solid rgba(251,191,36,0.2); }

/* Metric cards */
.metric-row { display: flex; gap: 0.85rem; margin-bottom: 1.6rem; flex-wrap: wrap; }
.metric-card {
    background: rgba(8,18,32,0.9);
    border: 1px solid rgba(56,189,248,0.14);
    border-radius: 13px;
    padding: 1.1rem 1.4rem;
    flex: 1; min-width: 130px;
    transition: all 0.25s;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, rgba(56,189,248,0.5), transparent);
}
.metric-card:hover { border-color: rgba(56,189,248,0.35); transform: translateY(-2px); }
.metric-card.warn::before { background: linear-gradient(90deg, transparent, rgba(251,191,36,0.5), transparent); }
.metric-card.good::before { background: linear-gradient(90deg, transparent, rgba(52,211,153,0.5), transparent); }
.metric-label { font-size: 0.68rem; color: #475569; text-transform: uppercase; letter-spacing: 1.2px; font-weight: 600; margin-bottom: 0.3rem; }
.metric-value { font-size: 1.75rem; font-weight: 700; color: #38bdf8; font-family: 'Fira Code', monospace; line-height: 1; }
.metric-value.warn { color: #fbbf24; }
.metric-value.good { color: #34d399; }
.metric-unit { font-size: 0.75rem; color: #475569; margin-top: 0.2rem; }

/* Section headers */
.sh {
    font-size: 1rem; font-weight: 600; color: #7dd3fc;
    border-left: 3px solid #38bdf8;
    padding-left: 0.75rem;
    margin: 1.6rem 0 0.9rem 0;
    letter-spacing: 0.2px;
}

/* Sidebar widgets */
[data-testid="stSlider"] > div > div > div { background: #38bdf8 !important; }
.stSelectbox > div > div {
    background: rgba(6,15,30,0.95) !important;
    border-color: rgba(56,189,248,0.25) !important;
    color: #dce8f5 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #0c4a6e, #0e3a5c);
    color: #bae6fd !important;
    border: 1px solid rgba(56,189,248,0.35) !important;
    border-radius: 9px;
    font-family: 'Outfit', sans-serif; font-weight: 600;
    padding: 0.55rem 1.5rem;
    transition: all 0.2s; width: 100%;
    letter-spacing: 0.3px;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0e5c87, #0c4a6e) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(56,189,248,0.2);
}

[data-testid="stExpander"] {
    background: rgba(6,15,30,0.7);
    border: 1px solid rgba(56,189,248,0.1);
    border-radius: 11px;
}

/* Info box */
.info-box {
    background: rgba(56,189,248,0.05);
    border: 1px solid rgba(56,189,248,0.18);
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    font-size: 0.87rem; color: #7dd3fc; line-height: 1.65;
    margin-bottom: 1.2rem;
}
.warn-box {
    background: rgba(251,191,36,0.05);
    border: 1px solid rgba(251,191,36,0.2);
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    font-size: 0.87rem; color: #fcd34d; line-height: 1.65;
    margin-bottom: 1rem;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: rgba(8,18,32,0.8); border-radius: 10px; padding: 4px; border: 1px solid rgba(56,189,248,0.1); }
.stTabs [data-baseweb="tab"] { color: #64748b !important; font-weight: 500; }
.stTabs [aria-selected="true"] { background: rgba(56,189,248,0.15) !important; color: #38bdf8 !important; border-radius: 7px !important; }
</style>
""", unsafe_allow_html=True)

# ─── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-title">🔵 DBSCAN Clustering</div>
  <div class="hero-sub">Density-Based Spatial Clustering of Applications with Noise</div>
  <div class="hero-tags">
    <span class="tag tag-algo">Density-Based</span>
    <span class="tag tag-type">Unsupervised</span>
    <span class="tag tag-data">Synthetic Clusters Dataset · 750 samples · 8 features</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Load Data ─────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic_clusters.csv')
df_raw = pd.read_csv(DATA_PATH)
feature_cols = list(df_raw.columns)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ DBSCAN Configuration")
    st.markdown("---")
    st.success(f"Dataset: {df_raw.shape[0]} samples · {len(feature_cols)} features")

    st.markdown("---")
    st.markdown("**🔢 Hyperparameters**")

    eps = st.slider(
        "ε — Epsilon (Neighborhood Radius)",
        min_value=0.1, max_value=5.0, value=0.5, step=0.05,
        help="Max distance between two points to be considered neighbors. Core parameter — too small = many noise points; too large = one big cluster."
    )

    min_samples = st.slider(
        "min_samples (Core Point Threshold)",
        min_value=2, max_value=30, value=5, step=1,
        help="Min neighbors in radius to be a core point. Higher = more noise, fewer clusters."
    )

    metric = st.selectbox(
        "Distance Metric",
        ["euclidean", "manhattan", "cosine"],
        help="Distance function used to compute neighborhoods."
    )

    algorithm = st.selectbox(
        "Algorithm",
        ["auto", "ball_tree", "kd_tree", "brute"],
        help="Nearest-neighbor search algorithm."
    )

    leaf_size = st.slider(
        "Leaf Size (ball/kd tree)",
        min_value=10, max_value=100, value=30, step=10,
        help="Affects speed and memory of ball_tree / kd_tree."
    )

    p_norm = st.selectbox(
        "p (Minkowski Power)",
        [1, 2],
        index=1,
        help="p=1 → Manhattan, p=2 → Euclidean (when metric='minkowski')."
    )

    scaling = st.selectbox("Feature Scaling", ["StandardScaler", "None"])

    viz_features = st.multiselect(
        "Visualization Features (pick 2–3)",
        feature_cols,
        default=["feature_1", "feature_2"],
        help="Features used for the scatter plots."
    )
    if len(viz_features) < 2:
        viz_features = ["feature_1", "feature_2"]

    viz_dim = st.radio("Scatter Dimensions", ["2D", "3D"], horizontal=True)
    st.markdown("---")
    run_btn = st.button("▶ Run DBSCAN", use_container_width=True)

st.markdown("""
<div class="info-box">
💡 <b>DBSCAN identifies three types of points:</b>
&nbsp;&nbsp;• <b>Core points</b> — have ≥ min_samples neighbors within ε
&nbsp;&nbsp;• <b>Border points</b> — within ε of a core point but below threshold
&nbsp;&nbsp;• <b>Noise points</b> — not reachable from any core point (labeled <b>−1</b>)
</div>
""", unsafe_allow_html=True)

# ─── Core Computation ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_dbscan(eps, min_samples, metric, algorithm, leaf_size, p_norm, scaling):
    df = pd.read_csv(DATA_PATH)
    X = df.values.astype(float)
    if scaling == "StandardScaler":
        X = StandardScaler().fit_transform(X)
    db = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric=metric,
        algorithm=algorithm,
        leaf_size=leaf_size,
        p=p_norm,
    )
    labels = db.fit_predict(X)
    return X, labels

@st.cache_data(show_spinner=False)
def compute_knn_distances(scaling):
    df = pd.read_csv(DATA_PATH)
    X = df.values.astype(float)
    if scaling == "StandardScaler":
        X = StandardScaler().fit_transform(X)
    nbrs = NearestNeighbors(n_neighbors=5).fit(X)
    distances, _ = nbrs.kneighbors(X)
    k_dist = np.sort(distances[:, -1])[::-1]
    return k_dist

if run_btn or True:
    with st.spinner("Running DBSCAN..."):
        X_scaled, labels = run_dbscan(eps, min_samples, metric, algorithm, leaf_size, p_norm, scaling)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    noise_pct = n_noise / len(labels) * 100

    # Compute metrics only when there are ≥2 clusters and not all noise
    valid_mask = labels != -1
    sil_score = db_score = ch_score = None
    if n_clusters >= 2 and valid_mask.sum() > n_clusters:
        try:
            sil_score = silhouette_score(X_scaled[valid_mask], labels[valid_mask])
            db_score  = davies_bouldin_score(X_scaled[valid_mask], labels[valid_mask])
            ch_score  = calinski_harabasz_score(X_scaled[valid_mask], labels[valid_mask])
        except Exception:
            pass

    PLOTLY_THEME = dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(4,12,24,0.9)",
        font=dict(family="Outfit", color="#94a3b8"),
        margin=dict(l=40, r=20, t=50, b=40),
    )

    # ── Metric Cards ─────────────────────────────────────────────────────────
    sil_str  = f"{sil_score:.3f}"  if sil_score is not None else "N/A"
    db_str   = f"{db_score:.3f}"   if db_score  is not None else "N/A"
    ch_str   = f"{ch_score:.0f}"   if ch_score  is not None else "N/A"
    noise_cls = "warn" if noise_pct > 20 else "good"

    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card good">
        <div class="metric-label">Clusters Found</div>
        <div class="metric-value good">{n_clusters}</div>
        <div class="metric-unit">excl. noise</div>
      </div>
      <div class="metric-card {noise_cls}">
        <div class="metric-label">Noise Points</div>
        <div class="metric-value {noise_cls}">{n_noise}</div>
        <div class="metric-unit">{noise_pct:.1f}% of data</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Silhouette Score</div>
        <div class="metric-value">{sil_str}</div>
        <div class="metric-unit">−1 to 1, higher = better</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Davies-Bouldin</div>
        <div class="metric-value">{db_str}</div>
        <div class="metric-unit">lower = better</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Calinski-Harabasz</div>
        <div class="metric-value">{ch_str}</div>
        <div class="metric-unit">higher = better</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">ε / min_samples</div>
        <div class="metric-value">{eps}</div>
        <div class="metric-unit">min_samples = {min_samples}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if n_clusters == 0:
        st.markdown('<div class="warn-box">⚠️ No clusters found — try increasing ε or decreasing min_samples.</div>', unsafe_allow_html=True)
    elif n_clusters == 1:
        st.markdown('<div class="warn-box">⚠️ Only 1 cluster found — try decreasing ε or increasing min_samples.</div>', unsafe_allow_html=True)

    # Cluster color palette — noise always gray
    unique_labels = sorted(set(labels))
    palette = px.colors.qualitative.Bold
    color_map = {
        str(lbl): ("#6b7280" if lbl == -1 else palette[i % len(palette)])
        for i, lbl in enumerate(unique_labels)
    }
    label_strs = [str(l) for l in labels]
    label_names = {str(l): ("Noise" if l == -1 else f"Cluster {l}") for l in unique_labels}
    display_labels = [label_names[s] for s in label_strs]
    color_map_display = {label_names[k]: v for k, v in color_map.items()}

    # ── Tabs for main plots ────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📍 Cluster Scatter", "📡 k-NN Distance Plot", "📊 Cluster Analysis", "🔬 PCA Projection"])

    with tab1:
        st.markdown(f'<div class="sh">Cluster Scatter — {viz_dim}</div>', unsafe_allow_html=True)
        viz_cols = viz_features if len(viz_features) >= 2 else ["feature_1", "feature_2"]
        col_idx = [feature_cols.index(c) for c in viz_cols[:3]]

        if viz_dim == "2D" or len(viz_cols) < 3:
            fig_sc = px.scatter(
                x=X_scaled[:, col_idx[0]],
                y=X_scaled[:, col_idx[1]],
                color=display_labels,
                color_discrete_map=color_map_display,
                labels={"x": viz_cols[0], "y": viz_cols[1], "color": "Cluster"},
                opacity=0.82,
            )
            fig_sc.update_traces(marker=dict(size=7, line=dict(width=0.4, color="rgba(0,0,0,0.4)")))
        else:
            fig_sc = px.scatter_3d(
                x=X_scaled[:, col_idx[0]],
                y=X_scaled[:, col_idx[1]],
                z=X_scaled[:, col_idx[2]],
                color=display_labels,
                color_discrete_map=color_map_display,
                labels={"x": viz_cols[0], "y": viz_cols[1], "z": viz_cols[2], "color": "Cluster"},
                opacity=0.82,
            )
            fig_sc.update_traces(marker=dict(size=4))
        fig_sc.update_layout(**PLOTLY_THEME, height=500)
        st.plotly_chart(fig_sc, use_container_width=True)

    with tab2:
        st.markdown('<div class="sh">k-NN Distance Plot (k=5) — Optimal ε Guide</div>', unsafe_allow_html=True)
        with st.spinner("Computing k-NN distances..."):
            k_dist = compute_knn_distances(scaling)

        fig_knn = go.Figure()
        fig_knn.add_scatter(
            y=k_dist, mode="lines",
            line=dict(color="#38bdf8", width=1.5),
            name="5-NN Distance",
        )
        fig_knn.add_hline(
            y=eps, line=dict(color="#fbbf24", dash="dash", width=1.5),
            annotation_text=f"Current ε = {eps}",
            annotation_font=dict(color="#fbbf24", size=11),
        )
        fig_knn.update_layout(
            **PLOTLY_THEME,
            xaxis_title="Points (sorted by distance desc)",
            yaxis_title="5th Nearest Neighbor Distance",
            title=dict(text="Look for the 'elbow' — that's your optimal ε", font=dict(size=12, color="#7dd3fc")),
            height=380,
        )
        st.plotly_chart(fig_knn, use_container_width=True)
        st.markdown("""
        <div class="info-box">
        📐 <b>How to use:</b> Find the "elbow" (sharp bend) in the curve — the ε value at that point
        is typically a good choice. Set ε in the sidebar to match.
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<div class="sh">Cluster Size Distribution</div>', unsafe_allow_html=True)
            label_series = pd.Series(display_labels)
            counts = label_series.value_counts().reset_index()
            counts.columns = ["Cluster", "Count"]
            counts["Color"] = counts["Cluster"].map(color_map_display)
            fig_bar = go.Figure(go.Bar(
                x=counts["Cluster"], y=counts["Count"],
                marker_color=counts["Color"].tolist(),
                marker_line_width=0,
                opacity=0.88,
            ))
            fig_bar.update_layout(**PLOTLY_THEME, xaxis_title="Cluster", yaxis_title="# Points", height=320)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_b:
            st.markdown('<div class="sh">Cluster Proportion</div>', unsafe_allow_html=True)
            fig_pie = go.Figure(go.Pie(
                labels=counts["Cluster"].tolist(),
                values=counts["Count"].tolist(),
                marker=dict(colors=counts["Color"].tolist()),
                hole=0.52,
                textinfo="label+percent",
                textfont=dict(size=11),
            ))
            fig_pie.update_layout(**PLOTLY_THEME, height=320, showlegend=False,
                                   margin=dict(l=10,r=10,t=20,b=10))
            st.plotly_chart(fig_pie, use_container_width=True)

        # Feature stats per cluster
        st.markdown('<div class="sh">Feature Statistics per Cluster</div>', unsafe_allow_html=True)
        df_labeled = df_raw.copy()
        df_labeled["Cluster"] = display_labels
        cluster_stats = df_labeled.groupby("Cluster")[feature_cols[:4]].agg(["mean", "std"]).round(3)
        st.dataframe(cluster_stats, use_container_width=True)

        # Density heatmap (feature1 vs feature2)
        st.markdown('<div class="sh">Density Heatmap (feature_1 vs feature_2)</div>', unsafe_allow_html=True)
        fig_dens = go.Figure(go.Histogram2dContour(
            x=X_scaled[:, 0], y=X_scaled[:, 1],
            colorscale="Blues", reversescale=False,
            contours=dict(showlabels=False),
            line=dict(width=0),
            ncontours=20,
            opacity=0.85,
        ))
        fig_dens.add_scatter(
            x=X_scaled[:, 0], y=X_scaled[:, 1],
            mode="markers",
            marker=dict(size=3, color=[color_map_display.get(d, "#6b7280") for d in display_labels], opacity=0.5),
            showlegend=False,
        )
        fig_dens.update_layout(**PLOTLY_THEME, xaxis_title="feature_1 (scaled)",
                                yaxis_title="feature_2 (scaled)", height=360)
        st.plotly_chart(fig_dens, use_container_width=True)

    with tab4:
        st.markdown('<div class="sh">PCA Projection (2D) — All 8 Features</div>', unsafe_allow_html=True)
        pca2 = PCA(n_components=2, random_state=42)
        X_pca2 = pca2.fit_transform(X_scaled)
        fig_pca = px.scatter(
            x=X_pca2[:, 0], y=X_pca2[:, 1],
            color=display_labels,
            color_discrete_map=color_map_display,
            labels={"x": f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)",
                    "y": f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)",
                    "color": "Cluster"},
            opacity=0.82,
        )
        fig_pca.update_traces(marker=dict(size=6, line=dict(width=0.3, color="rgba(0,0,0,0.3)")))
        fig_pca.update_layout(**PLOTLY_THEME, height=460,
                               title=dict(text="DBSCAN clusters in PCA space",
                                          font=dict(size=12, color="#7dd3fc")))
        st.plotly_chart(fig_pca, use_container_width=True)

        # Core / border / noise breakdown
        st.markdown('<div class="sh">Point Type Breakdown</div>', unsafe_allow_html=True)
        db_obj = DBSCAN(eps=eps, min_samples=min_samples, metric=metric,
                        algorithm=algorithm, leaf_size=leaf_size, p=p_norm)
        db_obj.fit(X_scaled)
        core_mask   = np.zeros(len(X_scaled), dtype=bool)
        core_mask[db_obj.core_sample_indices_] = True
        noise_mask  = labels == -1
        border_mask = ~core_mask & ~noise_mask

        type_counts = pd.DataFrame({
            "Type": ["Core Points", "Border Points", "Noise Points"],
            "Count": [core_mask.sum(), border_mask.sum(), noise_mask.sum()],
            "Color": ["#34d399", "#fbbf24", "#6b7280"],
        })
        fig_types = go.Figure(go.Bar(
            x=type_counts["Type"], y=type_counts["Count"],
            marker_color=type_counts["Color"].tolist(),
            marker_line_width=0, opacity=0.88,
        ))
        fig_types.update_layout(**PLOTLY_THEME, yaxis_title="Count", height=300)
        st.plotly_chart(fig_types, use_container_width=True)

    # ── Summary & Raw Data ────────────────────────────────────────────────────
    with st.expander("⚙️ Hyperparameter Summary"):
        hp_df = pd.DataFrame({
            "Parameter": ["ε (epsilon)", "min_samples", "metric", "algorithm",
                          "leaf_size", "p (Minkowski)", "scaling",
                          "Clusters Found", "Noise Points", "Silhouette"],
            "Value": [eps, min_samples, metric, algorithm,
                      leaf_size, p_norm, scaling,
                      n_clusters, n_noise, sil_str]
        })
        st.dataframe(hp_df, hide_index=True, use_container_width=True)

    with st.expander("📋 Labeled Data Preview (first 50 rows)"):
        preview = df_raw.head(50).copy()
        preview["Cluster Label"] = display_labels[:50]
        preview["Raw Label"]     = labels[:50]
        st.dataframe(preview, use_container_width=True)
