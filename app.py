"""
app.py — Fake Engagement Detection Dashboard
Streamlit frontend for the Hybrid Isolation Forest + Random Forest model.

Deploy via: streamlit run app.py
"""

import json
import os
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st


warnings.filterwarnings("ignore")

# ── Resolve all paths relative to this file's location ───────────────────────
BASE_DIR    = Path(__file__).resolve().parent
MODELS_DIR  = BASE_DIR          # .pkl files are in the repo root, not models/
OUTPUTS_DIR = BASE_DIR / "outputs"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake Engagement Detector",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Main background */
    .stApp { background-color: #0F1117; color: #E0E0E0; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #1A1D27; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1E2130, #252A3D);
        border: 1px solid #2E3250;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    div[data-testid="metric-container"] label { color: #8B9DC3 !important; font-size: 0.8rem !important; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #FFFFFF !important; font-size: 2rem !important; font-weight: 700 !important;
    }

    /* Section headers */
    h2 { color: #7B9FE0 !important; border-bottom: 1px solid #2E3250; padding-bottom: 8px; }
    h3 { color: #A8C0E8 !important; }

    /* Prediction result box */
    .result-bot {
        background: linear-gradient(135deg, #3D1A1A, #2D1010);
        border: 2px solid #FF4B4B;
        border-radius: 14px;
        padding: 20px 28px;
        text-align: center;
    }
    .result-organic {
        background: linear-gradient(135deg, #1A3D2A, #0E2718);
        border: 2px solid #21C55D;
        border-radius: 14px;
        padding: 20px 28px;
        text-align: center;
    }
    .result-title { font-size: 1.6rem; font-weight: 800; margin-bottom: 6px; }
    .result-sub   { font-size: 0.95rem; color: #A0A8B8; }

    /* Tab styling */
    button[data-baseweb="tab"] { color: #8B9DC3 !important; }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #7B9FE0 !important;
        border-bottom: 2px solid #7B9FE0 !important;
    }

    /* DataFrame */
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    /* Slider label */
    .stSlider label { color: #8B9DC3 !important; }

    /* Info/warning boxes */
    .stAlert { border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helpers ───────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "timing_regularity_score",
    "engagement_burst_ratio",
    "comment_similarity_score",
    "interaction_density_score",
    "follower_following_ratio",
    "posting_frequency",
    "behavioral_volatility_index",
]

FEATURE_LABELS = {
    "timing_regularity_score":   "Timing Regularity Score",
    "engagement_burst_ratio":    "Engagement Burst Ratio",
    "comment_similarity_score":  "Comment Similarity Score",
    "interaction_density_score": "Interaction Density Score",
    "follower_following_ratio":  "Follower / Following Ratio",
    "posting_frequency":         "Posting Frequency (posts/day)",
    "behavioral_volatility_index": "Behavioral Volatility Index",
}

FEATURE_HELP = {
    "timing_regularity_score":   "How regular and predictable the account's posting intervals are (0 = erratic, 1 = perfectly regular).",
    "engagement_burst_ratio":    "Proportion of engagements that occur in sudden, coordinated bursts.",
    "comment_similarity_score":  "Semantic similarity across comments made by this account (higher = copy-pasted / templated comments).",
    "interaction_density_score": "Volume of interactions relative to account age.",
    "follower_following_ratio":  "Ratio of followers to accounts being followed. Bots often follow many but attract few.",
    "posting_frequency":         "Average number of posts per day.",
    "behavioral_volatility_index": "Variance in the account's behaviour over time (lower = more bot-like).",
}


@st.cache_resource(show_spinner="Loading models…")
def load_models():
    """Load all serialised model artefacts from the models/ directory."""
    try:
        rf        = pickle.load(open(MODELS_DIR / "rf_model.pkl",       "rb"))
        iso       = pickle.load(open(MODELS_DIR / "iso_forest.pkl",     "rb"))
        scaler    = pickle.load(open(MODELS_DIR / "scaler.pkl",         "rb"))
        explainer = pickle.load(open(MODELS_DIR / "shap_explainer.pkl", "rb"))
        return rf, iso, scaler, explainer
    except FileNotFoundError as e:
        st.error(
            f"**Model files not found.** Make sure the `models/` directory "
            f"exists and contains the `.pkl` files.\n\n`{e}`"
        )
        st.stop()


def _find_file(filename):
    """Search for a file in repo root first, then outputs/ subfolder."""
    for candidate in [BASE_DIR / filename, OUTPUTS_DIR / filename]:
        if candidate.exists():
            return candidate
    return None


@st.cache_data
def load_metrics():
    path = _find_file("metrics.json")
    if path:
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data
def load_feature_importance():
    path = _find_file("feature_importance.csv")
    if path:
        return pd.read_csv(path)
    return None


def predict_single(features: dict, rf, iso, scaler):
    """Run inference on a single account feature dictionary."""
    X = np.array([[features[c] for c in FEATURE_COLS]])
    X_scaled = scaler.transform(X)

    label    = rf.predict(X_scaled)[0]
    proba    = rf.predict_proba(X_scaled)[0]          # [P(organic), P(bot)]
    iso_score = iso.decision_function(X_scaled)[0]    # higher = more normal
    iso_flag  = iso.predict(X_scaled)[0]              # -1 = anomaly

    return {
        "label":     label,
        "p_organic": proba[0],
        "p_bot":     proba[1],
        "iso_score": iso_score,
        "iso_flag":  iso_flag,
    }


def shap_bar_fig(shap_vals_1d, title="SHAP Feature Contributions"):
    """Horizontal bar chart of SHAP values for a single prediction."""
    pairs = sorted(zip(FEATURE_COLS, shap_vals_1d), key=lambda x: x[1])
    features, values = zip(*pairs)
    labels = [FEATURE_LABELS[f] for f in features]
    colors = ["#FF4B4B" if v > 0 else "#21C55D" for v in values]

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#1E2130")
    ax.set_facecolor("#1E2130")
    bars = ax.barh(labels, values, color=colors, height=0.6, edgecolor="#2E3250")
    ax.axvline(0, color="#5A6080", linewidth=1, linestyle="--")
    ax.set_xlabel("SHAP Value (impact on prediction)", color="#8B9DC3", fontsize=9)
    ax.set_title(title, color="#A8C0E8", fontsize=11, fontweight="bold")
    ax.tick_params(colors="#8B9DC3", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2E3250")
    for bar, val in zip(bars, values):
        ax.text(
            val + (0.005 if val >= 0 else -0.005),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}",
            va="center",
            ha="left" if val >= 0 else "right",
            color="#E0E0E0",
            fontsize=8,
        )
    plt.tight_layout()
    return fig






# ── Load artefacts ─────────────────────────────────────────────────────────────
rf, iso, scaler, explainer = load_models()
metrics = load_metrics()
fi_df   = load_feature_importance()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 Fake Engagement\nDetection System")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["📊 Overview", "🔍 Single Prediction", "📁 Batch Analysis", "📈 Model Insights"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        """
        **Model Architecture**
        - Isolation Forest (anomaly)
        - Random Forest (classifier)
        - SHAP (explainability)

        """
    )
    st.markdown("---")
    st.caption("Built with Streamlit · scikit-learn · SHAP")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 Model Performance Overview")
    st.markdown(
        "This dashboard presents the results of a hybrid ML pipeline trained to detect "
        "fake/bot social media accounts. The system combines unsupervised anomaly detection "
        "with supervised classification and SHAP-based explainability."
    )

    # ── KPI metrics ──────────────────────────────────────────────────────────
    if metrics:
        cm = metrics["confusion_matrix"]
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall    = tp / (tp + fn) if (tp + fn) else 0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy",  f"{metrics['accuracy']*100:.1f}%")
        col2.metric("ROC-AUC",   f"{metrics['roc_auc']:.4f}")
        col3.metric("Precision (Bot)", f"{precision:.2%}")
        col4.metric("Recall (Bot)",    f"{recall:.2%}")
        col5.metric("F1-Score (Bot)",  f"{f1:.2%}")

    # ── SHAP summary image ───────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("SHAP Beeswarm Summary")
    _shap_ov = _find_file("shap_summary.png")
    if _shap_ov:
        _, img_col, _ = st.columns([0.5, 3, 0.5])
        with img_col:
            st.image(str(_shap_ov), use_column_width=True)
        st.caption(
            "Each point represents one test sample. Colour indicates feature value "
            "(red = high, blue = low). Position on the x-axis shows the impact on the "
            "model's bot-classification output."
        )
        with st.expander("How to read this plot"):
            st.markdown(
                """
| Element | Meaning |
|---|---|
| **X-axis position** | Impact on the bot-classification output (positive = pushes toward Bot) |
| **Point colour (red)** | High feature value for that sample |
| **Point colour (blue)** | Low feature value for that sample |
| **Vertical spread** | Multiple samples with similar SHAP values |

**Example interpretation:** For `behavioral_volatility_index`, blue points (low volatility) appear on the right (positive SHAP = pushes toward Bot). This confirms that accounts with low behavioural variability are more likely to be classified as bots.
                """
            )
    else:
        st.info("Upload `shap_summary.png` to your repository.")




# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SINGLE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Single Prediction":
    st.title("🔍 Single Account Prediction")
    st.markdown(
        "Adjust the sliders below to enter an account's behavioural metrics. "
        "The model will classify it as **Organic** or **Bot** and provide a "
        "feature-level SHAP explanation."
    )

    # ── Input sliders ─────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    sliders = {}

    slider_configs = [
        ("timing_regularity_score",   0.0, 1.0,  0.3,  0.01),
        ("engagement_burst_ratio",    0.0, 1.0,  0.25, 0.01),
        ("comment_similarity_score",  0.0, 1.0,  0.28, 0.01),
        ("interaction_density_score", 0.0, 1.0,  0.22, 0.01),
        ("follower_following_ratio",  0.0, 10.0, 1.1,  0.1),
        ("posting_frequency",         0.0, 80.0, 5.0,  0.5),
        ("behavioral_volatility_index", 0.0, 1.0, 0.45, 0.01),
    ]

    for i, (feat, lo, hi, default, step) in enumerate(slider_configs):
        col = col_a if i < 4 else col_b
        with col:
            sliders[feat] = st.slider(
                FEATURE_LABELS[feat],
                min_value=float(lo),
                max_value=float(hi),
                value=float(default),
                step=float(step),
                help=FEATURE_HELP[feat],
            )

    st.markdown("")
    run = st.button("🚀 Classify Account", type="primary", use_container_width=True)

    if run:
        result = predict_single(sliders, rf, iso, scaler)

        st.markdown("---")
        res_col, explain_col = st.columns([1, 1.6])

        # ── Prediction result card ────────────────────────────────────────────
        with res_col:
            if result["label"] == 1:
                st.markdown(
                    f"""
                    <div class="result-bot">
                        <div class="result-title" style="color:#FF4B4B;">🤖 BOT DETECTED</div>
                        <div class="result-sub">Bot Probability: <strong style="color:#FF4B4B;">{result['p_bot']:.1%}</strong></div>
                        <div class="result-sub">Organic Probability: {result['p_organic']:.1%}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-organic">
                        <div class="result-title" style="color:#21C55D;">✅ ORGANIC ACCOUNT</div>
                        <div class="result-sub">Organic Probability: <strong style="color:#21C55D;">{result['p_organic']:.1%}</strong></div>
                        <div class="result-sub">Bot Probability: {result['p_bot']:.1%}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("")

            # Probability gauge
            st.markdown("**Confidence Breakdown**")
            prob_df = pd.DataFrame({
                "Class":       ["Organic", "Bot"],
                "Probability": [result["p_organic"], result["p_bot"]],
            })
            st.bar_chart(prob_df.set_index("Class"), color=["#21C55D"])

            # Isolation Forest signal
            iso_label = "⚠️ Anomaly" if result["iso_flag"] == -1 else "✅ Normal"
            st.metric(
                "Isolation Forest Signal",
                iso_label,
                help="Isolation Forest independently flags whether this account looks anomalous.",
            )
            st.metric(
                "Anomaly Score",
                f"{result['iso_score']:.4f}",
                help="Higher values indicate more typical / normal behaviour.",
            )

        # ── SHAP explanation ─────────────────────────────────────────────────
        with explain_col:
            st.markdown("**SHAP Feature Contributions**")
            X_input = np.array([[sliders[c] for c in FEATURE_COLS]])
            X_scaled = scaler.transform(X_input)
            sv = explainer.shap_values(X_scaled)

            if isinstance(sv, list):
                shap_1d = sv[1][0]
            elif sv.ndim == 3:
                shap_1d = sv[0, :, 1]
            else:
                shap_1d = sv[0]

            _shap_fig = shap_bar_fig(shap_1d, title="Feature Contributions to Bot Prediction")
            st.pyplot(_shap_fig)
            plt.close(_shap_fig)
            st.caption(
                "🔴 **Red bars** push the prediction toward *Bot*. "
                "🟢 **Green bars** push toward *Organic*. "
                "Bar length = magnitude of influence."
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — BATCH ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📁 Batch Analysis":
    st.title("📁 Batch Account Analysis")
    st.markdown(
        "Upload a CSV file containing multiple account records. The model will "
        "classify each row and return a results table with predictions and probabilities."
    )

    st.info(
        f"**Required columns:** `{'`, `'.join(FEATURE_COLS)}`",
        icon="ℹ️",
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        try:
            df_up = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not parse the uploaded file: `{e}`")
            st.stop()

        missing = [c for c in FEATURE_COLS if c not in df_up.columns]
        if missing:
            st.error(f"Missing required columns: `{missing}`")
            st.stop()

        X = df_up[FEATURE_COLS].values
        X_scaled = scaler.transform(X)

        labels   = rf.predict(X_scaled)
        probas   = rf.predict_proba(X_scaled)
        iso_flags = iso.predict(X_scaled)

        df_results = df_up.copy()
        df_results["prediction"]       = np.where(labels == 1, "Bot", "Organic")
        df_results["p_bot"]            = probas[:, 1].round(4)
        df_results["p_organic"]        = probas[:, 0].round(4)
        df_results["isolation_signal"] = np.where(iso_flags == -1, "Anomaly", "Normal")

        # ── Summary stats ─────────────────────────────────────────────────────
        n_bot  = (labels == 1).sum()
        n_org  = (labels == 0).sum()
        n_anom = (iso_flags == -1).sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Accounts",  len(df_results))
        c2.metric("🤖 Bots Detected", n_bot,  f"{n_bot/len(labels):.1%}")
        c3.metric("✅ Organic",       n_org,  f"{n_org/len(labels):.1%}")
        c4.metric("⚠️ IF Anomalies",  n_anom, f"{n_anom/len(labels):.1%}")

        st.markdown("---")

        # ── Donut chart ───────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(4, 3))
        fig.patch.set_facecolor("#1E2130")
        ax.set_facecolor("#1E2130")
        ax.pie(
            [n_org, n_bot],
            labels=["Organic", "Bot"],
            colors=["#21C55D", "#FF4B4B"],
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={"edgecolor": "#1E2130", "linewidth": 2},
            textprops={"color": "#E0E0E0"},
        )
        ax.set_title("Prediction Distribution", color="#A8C0E8", fontsize=11, fontweight="bold")
        col_chart, col_table = st.columns([1, 2])
        with col_chart:
            st.pyplot(fig)
            plt.close(fig)
        with col_table:
            st.markdown("**Results Table**")
            # Colour-code the prediction column
            def highlight_pred(val):
                if val == "Bot":
                    return "background-color: #3D1A1A; color: #FF4B4B; font-weight: bold"
                return "background-color: #1A3D2A; color: #21C55D; font-weight: bold"

            styled = df_results.style.applymap(
                highlight_pred, subset=["prediction"]
            ).format({"p_bot": "{:.2%}", "p_organic": "{:.2%}"})
            st.dataframe(styled, use_container_width=True, height=280)

        # ── Download ──────────────────────────────────────────────────────────
        st.download_button(
            label="⬇️  Download Results CSV",
            data=df_results.to_csv(index=False).encode(),
            file_name="batch_predictions.csv",
            mime="text/csv",
            type="primary",
        )
    else:
        # ── Template download ─────────────────────────────────────────────────
        st.markdown("#### No file uploaded yet.")
        template = pd.DataFrame(columns=FEATURE_COLS)
        st.download_button(
            label="⬇️  Download CSV Template",
            data=template.to_csv(index=False).encode(),
            file_name="template.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Insights":
    st.title("📈 Model Insights")

    tab_shap, tab_data = st.tabs(["📊 SHAP Analysis", "🗂️ Dataset Explorer"])

    # ── Tab 1 — Combined SHAP Analysis ───────────────────────────────────────
    with tab_shap:
        st.subheader("SHAP Feature Importance")
        st.markdown(
            "Features are ranked by their **mean absolute SHAP value** — "
            "a measure of each feature's average contribution to the model's predictions across the entire test set. "
            "The higher the value, the more influence that feature has on whether the model classifies an account as Bot or Organic."
        )
        if fi_df is None:
            st.error(
                "⚠️ `feature_importance.csv` could not be found. "
                f"Searched in: `{BASE_DIR}` and `{OUTPUTS_DIR}`. "
                "Please ensure the file exists in either the repo root or the `outputs/` subfolder."
            )
        if fi_df is not None:
            # Bar chart
            fig_fi, ax_fi = plt.subplots(figsize=(7, 4))
            fig_fi.patch.set_facecolor("#1E2130")
            ax_fi.set_facecolor("#1E2130")
            labels_fi = [FEATURE_LABELS[f] for f in fi_df["feature"]][::-1]
            ax_fi.barh(labels_fi, fi_df["importance"].values[::-1],
                       color="#7B9FE0", height=0.6, edgecolor="#2E3250")
            ax_fi.set_xlabel("Mean |SHAP Value|", color="#8B9DC3", fontsize=9)
            ax_fi.tick_params(colors="#8B9DC3", labelsize=8)
            for spine in ax_fi.spines.values():
                spine.set_edgecolor("#2E3250")
            plt.tight_layout()
            st.pyplot(fig_fi)
            plt.close(fig_fi)

            # Ranked table
            fi_display = fi_df.copy()
            fi_display["Rank"]        = range(1, len(fi_df) + 1)
            fi_display["Feature"]     = fi_df["feature"].map(FEATURE_LABELS)
            fi_display["Mean |SHAP|"] = fi_df["importance"].round(4)
            fi_display["Relative %"]  = (fi_df["importance"] / fi_df["importance"].sum() * 100).round(1)
            st.dataframe(
                fi_display[["Rank", "Feature", "Mean |SHAP|", "Relative %"]],
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("---")
        st.subheader("Feature-by-Feature SHAP Interpretation")
        st.markdown(
            "Each card below provides a complete SHAP interpretation for one feature. "
            "It explains: **(1) what pattern appears in the beeswarm plot**, **(2) what that pattern means "
            "in plain language**, and **(3) why this matters** for detecting bot accounts. "
            "SHAP values and rankings update automatically if the model is retrained."
        )

        fi_desc = {
            "behavioral_volatility_index": {
                "beeswarm": (
                    "🔵 **Blue dots** (accounts with LOW volatility) appear **far to the right** of zero — "
                    "these are the strongest pushes toward a Bot classification. "
                    "🔴 **Red dots** (accounts with HIGH volatility) appear **to the left** of zero — "
                    "pushing toward Organic. The spread is the widest of all seven features, confirming this is the dominant predictor."
                ),
                "meaning": (
                    "Behavioural volatility measures how much an account's activity varies over time. "
                    "**Bots operate like scheduled scripts** — they post at fixed intervals, like the same number of "
                    "posts per hour, every hour. This produces near-zero variance. "
                    "Human users, by contrast, are active some days and quiet others, post more in the evenings, "
                    "and have unpredictable bursts of activity — all of which produce high volatility."
                ),
                "shap_insight": (
                    "Because this feature has the **largest mean |SHAP| value (35% of total)**, "
                    "it is the single most influential input the Random Forest relies on. "
                    "When volatility is low, it almost always overrides all other features and forces a Bot prediction. "
                    "This also explains why it is placed at the top of the beeswarm plot — SHAP orders features by importance."
                ),
            },
            "comment_similarity_score": {
                "beeswarm": (
                    "🔴 **Red dots** (HIGH similarity) cluster tightly **to the right** — pushing toward Bot. "
                    "🔵 **Blue dots** (LOW similarity) spread to the **left** toward Organic. "
                    "The separation is clear, making this a reliable directional signal."
                ),
                "meaning": (
                    "Comment similarity measures how semantically alike an account's comments are across different posts. "
                    "**Bots reuse templated, copy-pasted messages** — promotional text, spam phrases, or "
                    "near-identical replies posted repeatedly across many posts. "
                    "Organic users write different comments in different contexts — varied wording, tone, and content "
                    "— keeping this score low."
                ),
                "shap_insight": (
                    "Ranked 2nd at **27.7% of total SHAP importance**. "
                    "This feature is particularly valuable because it captures the *content* of bot behaviour, "
                    "not just the *timing*. Combined with behavioural volatility, these two features alone explain "
                    "over 60% of the model's decision-making."
                ),
            },
            "follower_following_ratio": {
                "beeswarm": (
                    "🔴 **Red dots** (HIGH ratio) are spread **far to the right** with a wide scatter — "
                    "a strong and consistent bot signal. "
                    "🔵 **Blue dots** (LOW ratio) cluster near zero with little influence. "
                    "The outlier dots extending past 0.4 SHAP are almost entirely bots with extreme ratios."
                ),
                "meaning": (
                    "This ratio compares the number of followers an account has to the number it follows. "
                    "**Bots aggressively follow large numbers of accounts** hoping to receive follow-backs, "
                    "which inflates their following count disproportionately relative to their actual followers. "
                    "Organic users typically maintain a balanced ratio close to 1."
                ),
                "shap_insight": (
                    "Despite showing **weak correlation (r < 0.37)** with all other features in the correlation analysis, "
                    "this ranks 3rd at **19.8% importance**. This is a key finding: the feature captures an "
                    "entirely **independent dimension** of bot behaviour that the other features do not overlap with. "
                    "Removing it would meaningfully reduce model accuracy."
                ),
            },
            "posting_frequency": {
                "beeswarm": (
                    "🔴 **Red dots** (HIGH frequency) lean **to the right** — pushing toward Bot. "
                    "🔵 **Blue dots** (LOW frequency) sit **left of zero** — pushing toward Organic. "
                    "The separation is moderate, not as clean as the top two features."
                ),
                "meaning": (
                    "Posting frequency measures the average number of posts per day. "
                    "**Bots post continuously and at scale** to maximise reach and engagement, "
                    "often posting dozens of times per day. "
                    "Organic users post occasionally — typically 1–5 times per day at most — "
                    "resulting in a much lower and more natural posting rate."
                ),
                "shap_insight": (
                    "Ranked 4th at **9.9% importance**. "
                    "Its contribution is smaller than expected because it **partially overlaps** with behavioural volatility "
                    "— a bot that posts frequently also tends to post at regular intervals, so the volatility feature "
                    "already captures much of the same signal."
                ),
            },
            "timing_regularity_score": {
                "beeswarm": (
                    "Points cluster **tightly near zero** for both classes. "
                    "A slight rightward lean for higher values is visible, but the effect is small. "
                    "There is no dramatic separation — most dots sit within ±0.1 SHAP."
                ),
                "meaning": (
                    "Timing regularity measures how predictably spaced an account's posts are. "
                    "A score of 1.0 means posts arrive at perfectly fixed intervals (like a cron job); "
                    "0.0 means completely random timing. "
                    "**Automated accounts post like scheduled tasks** — every 15 minutes, every hour — "
                    "producing very high regularity. "
                    "Human posting is inherently irregular."
                ),
                "shap_insight": (
                    "Ranked 5th at only **6.3% importance**. "
                    "Although the feature logically should distinguish bots from humans, "
                    "its SHAP contribution is small because **behavioural volatility already captures the same information** "
                    "more completely. This is a case of feature redundancy — "
                    "both features measure predictability of behaviour, so the model learns to rely on the stronger one."
                ),
            },
            "engagement_burst_ratio": {
                "beeswarm": (
                    "**Most points hug zero** — the majority of both bot and organic accounts have near-zero SHAP "
                    "contribution from this feature. "
                    "A small cluster of 🔴 **red outlier dots** extends **to the right**, "
                    "representing a specific subset of high-burst bots."
                ),
                "meaning": (
                    "Engagement burst ratio captures how much of an account's total engagement happens in "
                    "sudden, concentrated spikes rather than gradually. "
                    "**Coordinated bot networks** — where many bots simultaneously engage with the same post — "
                    "produce extreme burst patterns. "
                    "However, many simpler bots operate individually and do not produce bursts, "
                    "which is why this signal is inconsistent."
                ),
                "shap_insight": (
                    "Ranked 6th at **5.5% importance**. "
                    "The tight clustering near zero for most samples confirms this is a **narrow, specialised signal** — "
                    "useful for identifying coordinated bot networks specifically, "
                    "but not a reliable general-purpose bot indicator. "
                    "It contributes only when the burst ratio is unusually high."
                ),
            },
            "interaction_density_score": {
                "beeswarm": (
                    "**Extremely tight clustering near zero** for almost all samples in both classes. "
                    "The dots barely move from the centre line, indicating this feature has "
                    "almost no influence on individual predictions."
                ),
                "meaning": (
                    "Interaction density measures the volume of interactions (likes, comments, shares) "
                    "relative to the account's age. "
                    "In theory, bots should have unnaturally high interaction density. "
                    "In practice, this signal is **already fully explained** by posting frequency and comment similarity — "
                    "an account that posts frequently and with repetitive comments will naturally also have high interaction density."
                ),
                "shap_insight": (
                    "Ranked last at **3.3% importance** — the weakest contributor. "
                    "The correlation analysis confirmed this: interaction density has correlations above r=0.6 "
                    "with both posting frequency and comment similarity. "
                    "In machine learning terms, this feature is **collinear** with stronger predictors, "
                    "so the Random Forest assigns it almost no weight. "
                    "It could be removed from the feature set without meaningfully reducing model accuracy."
                ),
            },
        }
        rank_emojis = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣", "7️⃣"]
        if fi_df is not None:
            total_imp = fi_df["importance"].sum()
            for i, row in fi_df.reset_index(drop=True).iterrows():
                feat_key   = row["feature"]
                shap_val   = f"{row['importance']:.4f}"
                rel_pct    = f"{row['importance'] / total_imp * 100:.1f}%"
                feat_label = FEATURE_LABELS.get(feat_key, feat_key)
                rank_icon  = rank_emojis[i] if i < len(rank_emojis) else f"{i+1}."
                info       = fi_desc.get(feat_key, {})
                with st.expander(f"{rank_icon}  **{feat_label}** — Mean |SHAP| `{shap_val}` · `{rel_pct}` of total"):
                    st.markdown("**📈 Beeswarm Plot Pattern**")
                    st.markdown(info.get("beeswarm", ""))
                    st.markdown("**🧠 What This Feature Measures**")
                    st.markdown(info.get("meaning", ""))
                    st.markdown("**📊 SHAP Significance**")
                    st.markdown(info.get("shap_insight", ""))

    # ── Tab 2 — Dataset Explorer ──────────────────────────────────────────────
    with tab_data:
        st.subheader("Dataset Explorer")
        st.markdown(
            "Explore how each behavioural feature is distributed across Bot and Organic accounts. "
            "Select a feature to see how clearly it separates the two classes."
        )
        _csv_path = _find_file("social_media_dataset.csv")
        if _csv_path:
            df_data = pd.read_csv(_csv_path)

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Records",    len(df_data))
            c2.metric("Bot Accounts",     int((df_data["label"] == 1).sum()))
            c3.metric("Organic Accounts", int((df_data["label"] == 0).sum()))

            st.markdown("---")
            feat_sel = st.selectbox(
                "Select feature to visualise",
                FEATURE_COLS,
                format_func=lambda x: FEATURE_LABELS[x],
            )
            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor("#1E2130")
            ax.set_facecolor("#1E2130")
            bins = np.linspace(
                df_data[feat_sel].min(), df_data[feat_sel].max(), 41
            )
            # Filled bars for Organic (drawn first, behind)
            ax.hist(df_data[df_data["label"]==0][feat_sel], bins=bins,
                    alpha=0.55, color="#21C55D", label="Organic", edgecolor="none")
            # Outline-only bars for Bot (drawn on top so overlap is visible as outline)
            ax.hist(df_data[df_data["label"]==1][feat_sel], bins=bins,
                    alpha=0.75, color="#FF4B4B", label="Bot",
                    edgecolor="#FF4B4B", linewidth=0.8, histtype="stepfilled")
            ax.set_xlabel(FEATURE_LABELS[feat_sel], color="#8B9DC3", fontsize=9)
            ax.set_ylabel("Count", color="#8B9DC3", fontsize=9)
            ax.set_title(f"Distribution of {FEATURE_LABELS[feat_sel]}", color="#A8C0E8",
                         fontsize=11, fontweight="bold")
            ax.tick_params(colors="#8B9DC3")
            legend = ax.legend(facecolor="#252A3D", edgecolor="#2E3250", labelcolor="#E0E0E0")
            for spine in ax.spines.values():
                spine.set_edgecolor("#2E3250")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption(
                "🟢 **Green** = Organic accounts · 🔴 **Red** = Bot accounts · "
                "**Where green and red overlap**, the colours blend — this overlap zone indicates "
                "that both classes share similar values for this feature in that range, "
                "meaning the feature alone cannot distinguish them in that region."
            )

            dist_insights = {
                "timing_regularity_score": (
                    "**What the distribution reveals:** "
                    "Bot accounts (red) are shifted toward **higher regularity values** with a tighter, narrower spread — "
                    "confirming they post at predictable, clock-like intervals. "
                    "Organic users (green) show a broader distribution at lower values, reflecting natural irregular activity. "
                    "The **overlap zone** in the mid-range indicates some organic users also post fairly regularly "
                    "(e.g. scheduled content creators), which reduces this feature's reliability on its own. "
                    "This is why it ranks 5th in SHAP importance despite being logically intuitive."
                ),
                "engagement_burst_ratio": (
                    "**What the distribution reveals:** "
                    "Both classes **overlap heavily at lower values (0.0–0.6)** — both bots and organic users can have "
                    "low burst ratios, which is why this feature ranks 6th in SHAP importance. "
                    "The key signal is the **right tail (0.6–1.0)**: bots extend significantly further right "
                    "with extreme burst values that organic users rarely produce. "
                    "The large overlap zone confirms this feature is a narrow, specialised signal "
                    "— it only fires reliably for coordinated bot networks, not all bot types."
                ),
                "comment_similarity_score": (
                    "**What the distribution reveals:** "
                    "This is one of the **cleanest separations** in the dataset. "
                    "Bots cluster at **higher similarity scores** (copy-pasted, templated comments), "
                    "while organic users concentrate at **lower scores** (varied, context-specific writing). "
                    "The overlap zone is relatively narrow, appearing around mid-range values — "
                    "some organic users do repeat phrases occasionally, and some bots vary their messages slightly. "
                    "The clear separation confirms why this ranks 2nd in SHAP importance."
                ),
                "interaction_density_score": (
                    "**What the distribution reveals:** "
                    "The two distributions **heavily overlap throughout the entire range** — "
                    "there is almost no region where one class clearly dominates. "
                    "This directly explains why this feature ranks last in SHAP importance (3.3%): "
                    "the model cannot use it to reliably separate bots from organic accounts. "
                    "Its information is entirely captured by posting frequency and comment similarity, "
                    "which are much stronger predictors of the same underlying behaviour."
                ),
                "follower_following_ratio": (
                    "**What the distribution reveals:** "
                    "Organic users (green) concentrate near a **ratio of 1** (roughly equal followers and following). "
                    "Bots (red) extend further right with **higher ratios**, reflecting their aggressive "
                    "mass-following strategy. "
                    "The overlap zone near ratio 1–3 includes both classes, meaning a moderately high ratio alone "
                    "is not definitive. However, extreme ratios (above 5) are almost exclusively bots — "
                    "this is the region that contributes the most SHAP value for this feature."
                ),
                "posting_frequency": (
                    "**What the distribution reveals:** "
                    "Organic users (green) cluster at **low posting frequencies** (left side), "
                    "reflecting natural, occasional posting behaviour. "
                    "Bots (red) spread across **higher frequencies**, consistent with automated, high-volume posting. "
                    "The overlap at low-to-mid frequencies means casual bots that post infrequently "
                    "are harder to detect using this feature alone — "
                    "which is why it ranks 4th rather than higher despite being intuitively important."
                ),
                "behavioral_volatility_index": (
                    "**What the distribution reveals:** "
                    "This is the **clearest class separation of all seven features**. "
                    "Bots (red) cluster almost entirely at **very low volatility values** (scripted, predictable, repetitive). "
                    "Organic users (green) spread across **higher volatility values** (irregular, spontaneous, human). "
                    "The overlap zone is minimal — only the edges of each distribution touch. "
                    "This cleanest-possible separation is exactly why this feature holds the #1 SHAP rank (35% of total importance): "
                    "a low volatility score is the single strongest individual signal of automated bot behaviour."
                ),
            }
            st.info(dist_insights.get(feat_sel, ""))
        else:
            st.info(
                "Dataset not found. Run `generate_dataset.py` to create "
                "`social_media_dataset.csv` in the project root."
            )
