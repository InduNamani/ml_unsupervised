import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
import os

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Isolation Forest · Anomaly Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark background */
.stApp {
    background: #0a0e1a;
    color: #e2e8f0;
}

/* Main area */
.block-container {
    padding: 2rem 2.5rem 3rem;
    max-width: 1400px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f1526;
    border-right: 1px solid #1e2d4a;
}
section[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Space Mono', monospace;
    color: #38bdf8;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* Hero header */
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    color: #f0f9ff;
    letter-spacing: -0.03em;
    line-height: 1.1;
}
.hero-accent {
    color: #38bdf8;
}
.hero-sub {
    font-size: 1rem;
    color: #64748b;
    margin-top: 0.4rem;
    font-weight: 300;
}

/* Metric cards */
.metric-card {
    background: #111827;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
}
.metric-label {
    font-size: 0.7rem;
    font-family: 'Space Mono', monospace;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #f0f9ff;
    line-height: 1.2;
    margin-top: 0.2rem;
}
.metric-value.anomaly { color: #f87171; }
.metric-value.normal  { color: #34d399; }

/* Section headers */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #38bdf8;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e2d4a;
    margin-bottom: 1rem;
}

/* Divider */
hr { border-color: #1e2d4a; }

/* DataFrame */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* Slider labels */
.stSlider label { color: #94a3b8 !important; font-size: 0.85rem; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0369a1, #4f46e5);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.05em;
    padding: 0.55rem 1.4rem;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* Alerts */
.anomaly-alert {
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    color: #fca5a5;
    font-size: 0.9rem;
}
.success-box {
    background: rgba(52, 211, 153, 0.08);
    border: 1px solid rgba(52, 211, 153, 0.3);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    color: #6ee7b7;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ─── Helpers ─────────────────────────────────────────────────────────────────────
FEATURE_COLS = ["amount", "hour", "duration_sec", "num_items", "customer_age"]

@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "..", "data", "transactions.csv")
    return pd.read_csv(path)

def run_isolation_forest(df, n_estimators, contamination, max_samples, max_features, random_state):
    X = df[FEATURE_COLS].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples=max_samples,
        max_features=max_features,
        random_state=random_state,
        n_jobs=-1,
    )
    preds = model.fit_predict(X_scaled)
    scores = model.decision_function(X_scaled)

    df = df.copy()
    df["prediction"] = np.where(preds == -1, "anomaly", "normal")
    df["anomaly_score"] = -scores          # higher = more anomalous
    df["marker_size"] = np.clip(-scores, 0, None) + 2   # ensure ≥0 for plotly size
    df["raw_pred"] = preds

    # PCA for 2D visualisation
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)
    df["pca_1"] = pca_coords[:, 0]
    df["pca_2"] = pca_coords[:, 1]

    return df, model, scaler

PLOT_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#94a3b8"),
    xaxis=dict(gridcolor="#1e2d4a", zerolinecolor="#1e2d4a"),
    yaxis=dict(gridcolor="#1e2d4a", zerolinecolor="#1e2d4a"),
    margin=dict(l=10, r=10, t=40, b=10),
)
COLOR_MAP = {"normal": "#34d399", "anomaly": "#f87171"}

# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔧 Hyperparameters")
    st.markdown("---")

    n_estimators = st.slider(
        "n_estimators  *(# trees)*", 50, 500, 100, 10,
        help="Number of base estimators in the ensemble. More trees = more stable but slower."
    )
    contamination = st.slider(
        "contamination  *(expected anomaly %)*", 0.01, 0.30, 0.10, 0.01,
        help="Expected proportion of anomalies in the dataset."
    )
    max_samples_pct = st.slider(
        "max_samples  *(% of dataset)*", 10, 100, 100, 5,
        help="Number of samples drawn to train each tree (as % of total rows)."
    )
    max_features = st.slider(
        "max_features  *(# features per tree)*", 1, len(FEATURE_COLS), len(FEATURE_COLS), 1,
        help="Number of features drawn to train each base estimator."
    )
    random_state = st.number_input("random_state", 0, 9999, 42, 1)

    st.markdown("---")
    st.markdown("### 📊 Visualization")
    plot_feature_x = st.selectbox("X-axis feature", FEATURE_COLS, index=0)
    plot_feature_y = st.selectbox("Y-axis feature", FEATURE_COLS, index=2)

    run_btn = st.button("▶  Run Detection", use_container_width=True)

# ─── Main ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-title">
    Isolation<span class="hero-accent"> Forest</span>
</div>
<div class="hero-sub">Unsupervised Anomaly Detection · Transaction Dataset</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Load data
df_raw = load_data()

# Show dataset preview
with st.expander("📂  Dataset Preview", expanded=False):
    st.dataframe(df_raw.head(20), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.metric("Total Records", f"{len(df_raw):,}")
    c2.metric("Features", len(FEATURE_COLS))

# Run model on button or first load
if "result_df" not in st.session_state or run_btn:
    max_samples_val = max(1, int(len(df_raw) * max_samples_pct / 100))
    with st.spinner("Running Isolation Forest…"):
        result_df, model, scaler = run_isolation_forest(
            df_raw, n_estimators, contamination,
            max_samples_val, max_features, random_state
        )
    st.session_state["result_df"] = result_df

df = st.session_state["result_df"]

# ─── KPI Cards ───────────────────────────────────────────────────────────────────
n_anom = (df["prediction"] == "anomaly").sum()
n_norm = (df["prediction"] == "normal").sum()
avg_score_anom = df[df["prediction"] == "anomaly"]["anomaly_score"].mean()

# Ground-truth accuracy (if label column exists)
if "label" in df.columns:
    correct = (df["prediction"] == df["label"]).sum()
    accuracy = correct / len(df) * 100
else:
    accuracy = None

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Total Records</div>
        <div class="metric-value">{len(df):,}</div></div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Anomalies Found</div>
        <div class="metric-value anomaly">{n_anom}</div></div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Normal Records</div>
        <div class="metric-value normal">{n_norm}</div></div>""", unsafe_allow_html=True)
with col4:
    val = f"{accuracy:.1f}%" if accuracy is not None else "N/A"
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Accuracy vs Labels</div>
        <div class="metric-value">{val}</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Plots Row 1 ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📈  Detection Overview</div>', unsafe_allow_html=True)

col_a, col_b = st.columns([3, 2])

with col_a:
    # Scatter: selected features
    fig1 = px.scatter(
        df, x=plot_feature_x, y=plot_feature_y,
        color="prediction",
        color_discrete_map=COLOR_MAP,
        size="marker_size",
        size_max=14,
        opacity=0.8,
        hover_data=["transaction_id", "anomaly_score"],
        title=f"{plot_feature_x}  vs  {plot_feature_y}",
    )
    fig1.update_layout(**PLOT_THEME, legend_title="")
    fig1.update_traces(marker=dict(line=dict(width=0)))
    st.plotly_chart(fig1, use_container_width=True)

with col_b:
    # Anomaly score distribution
    fig2 = go.Figure()
    for label, color in COLOR_MAP.items():
        subset = df[df["prediction"] == label]["anomaly_score"]
        fig2.add_trace(go.Histogram(
            x=subset, name=label,
            marker_color=color,
            opacity=0.75,
            nbinsx=30,
        ))
    fig2.update_layout(
        **PLOT_THEME,
        title="Anomaly Score Distribution",
        barmode="overlay",
        legend_title="",
    )
    st.plotly_chart(fig2, use_container_width=True)

# ─── Plots Row 2 ─────────────────────────────────────────────────────────────────
col_c, col_d = st.columns([2, 3])

with col_c:
    # Pie / Donut
    counts = df["prediction"].value_counts()
    fig3 = go.Figure(go.Pie(
        labels=counts.index.tolist(),
        values=counts.values.tolist(),
        hole=0.55,
        marker=dict(colors=["#34d399", "#f87171"]),
        textinfo="label+percent",
        textfont=dict(size=13),
    ))
    fig3.update_layout(**PLOT_THEME, title="Prediction Breakdown", showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

with col_d:
    # PCA 2D scatter
    fig4 = px.scatter(
        df, x="pca_1", y="pca_2",
        color="prediction",
        color_discrete_map=COLOR_MAP,
        opacity=0.75,
        hover_data=["transaction_id", "anomaly_score"],
        title="PCA 2D Projection",
    )
    fig4.update_layout(**PLOT_THEME, legend_title="")
    fig4.update_traces(marker=dict(size=6, line=dict(width=0)))
    st.plotly_chart(fig4, use_container_width=True)

# ─── Plots Row 3 ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔬  Feature Analysis</div>', unsafe_allow_html=True)

# Box plots for each feature
fig5 = make_subplots(rows=1, cols=len(FEATURE_COLS),
                     subplot_titles=FEATURE_COLS)
for i, feat in enumerate(FEATURE_COLS, 1):
    for label, color in COLOR_MAP.items():
        subset = df[df["prediction"] == label][feat]
        fig5.add_trace(
            go.Box(y=subset, name=label, marker_color=color,
                   showlegend=(i == 1), boxmean=True),
            row=1, col=i,
        )
fig5.update_layout(
    **PLOT_THEME,
    title="Feature Distribution: Normal vs Anomaly",
    height=350,
    boxmode="group",
)
fig5.update_xaxes(showticklabels=False)
st.plotly_chart(fig5, use_container_width=True)

# Anomaly score over transaction index
fig6 = go.Figure()
fig6.add_trace(go.Scatter(
    x=df.index, y=df["anomaly_score"],
    mode="markers",
    marker=dict(
        color=df["anomaly_score"],
        colorscale=[[0, "#34d399"], [0.5, "#fbbf24"], [1, "#f87171"]],
        size=5,
        showscale=True,
        colorbar=dict(title="Score"),
    ),
    text=df["prediction"],
    hovertemplate="<b>Index %{x}</b><br>Score: %{y:.4f}<br>%{text}",
))
fig6.update_layout(**PLOT_THEME, title="Anomaly Score per Record (sorted by index)", height=250)
st.plotly_chart(fig6, use_container_width=True)

# ─── Confusion Matrix (if ground truth exists) ────────────────────────────────
if "label" in df.columns:
    st.markdown('<div class="section-header">🎯  Evaluation vs Ground Truth</div>', unsafe_allow_html=True)
    col_e, col_f = st.columns([2, 3])

    with col_e:
        labels_order = ["normal", "anomaly"]
        cm = confusion_matrix(df["label"], df["prediction"], labels=labels_order)
        fig7 = px.imshow(
            cm,
            x=labels_order, y=labels_order,
            color_continuous_scale=[[0, "#111827"], [1, "#38bdf8"]],
            text_auto=True,
            labels=dict(x="Predicted", y="Actual"),
            title="Confusion Matrix",
        )
        fig7.update_layout(**PLOT_THEME, coloraxis_showscale=False)
        fig7.update_traces(textfont_size=18)
        st.plotly_chart(fig7, use_container_width=True)

    with col_f:
        report = classification_report(df["label"], df["prediction"],
                                       labels=["normal", "anomaly"],
                                       output_dict=True)
        rep_df = pd.DataFrame(report).T.round(3).drop("support", axis=1, errors="ignore")
        st.markdown("**Classification Report**")
        st.dataframe(rep_df.style.format("{:.3f}").background_gradient(
            cmap="Blues", axis=None), use_container_width=True)

# ─── Anomaly Records Table ────────────────────────────────────────────────────
st.markdown('<div class="section-header">🚨  Detected Anomalies</div>', unsafe_allow_html=True)

anom_df = df[df["prediction"] == "anomaly"].sort_values("anomaly_score", ascending=False)
display_cols = ["transaction_id", "amount", "hour", "duration_sec",
                "num_items", "customer_age", "anomaly_score"]
if "label" in df.columns:
    display_cols.append("label")

st.markdown(f'<div class="anomaly-alert">⚠️  <strong>{len(anom_df)} anomalous transactions</strong> detected with current hyperparameters. Review table below.</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.dataframe(
    anom_df[display_cols].reset_index(drop=True)
        .style.format({"amount": "₹{:.2f}", "anomaly_score": "{:.4f}"}),
    use_container_width=True,
    height=300,
)

st.markdown("<br>", unsafe_allow_html=True)
st.download_button(
    "⬇  Download Anomaly Report (CSV)",
    data=anom_df[display_cols].to_csv(index=False).encode(),
    file_name="anomaly_report.csv",
    mime="text/csv",
)
