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
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

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


def roc_curve_fig(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#1E2130")
    ax.set_facecolor("#1E2130")
    ax.plot(fpr, tpr, color="#7B9FE0", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], color="#5A6080", lw=1, linestyle="--")
    ax.set_xlabel("False Positive Rate", color="#8B9DC3", fontsize=9)
    ax.set_ylabel("True Positive Rate", color="#8B9DC3", fontsize=9)
    ax.set_title("ROC Curve", color="#A8C0E8", fontsize=11, fontweight="bold")
    ax.tick_params(colors="#8B9DC3")
    ax.legend(facecolor="#252A3D", edgecolor="#2E3250", labelcolor="#E0E0E0", fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2E3250")
    plt.tight_layout()
    return fig


def confusion_matrix_fig(cm):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    fig.patch.set_facecolor("#1E2130")
    ax.set_facecolor("#1E2130")
    disp = ConfusionMatrixDisplay(confusion_matrix=np.array(cm),
                                   display_labels=["Organic", "Bot"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix", color="#A8C0E8", fontsize=11, fontweight="bold")
    ax.tick_params(colors="#8B9DC3")
    ax.xaxis.label.set_color("#8B9DC3")
    ax.yaxis.label.set_color("#8B9DC3")
    for text in disp.text_.ravel():
        text.set_color("#FFFFFF")
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

    # ── Confusion matrix + Feature importance (only if data available) ─────────
    if metrics or fi_df is not None:
        st.markdown("---")
        col_left, col_right = st.columns(2)

        with col_left:
            if metrics:
                st.pyplot(confusion_matrix_fig(metrics["confusion_matrix"]))
                st.caption(
                    f"True Negatives: **{tn}** | False Positives: **{fp}** | "
                    f"False Negatives: **{fn}** | True Positives: **{tp}**"
                )

        with col_right:
            if fi_df is not None:
                fig, ax = plt.subplots(figsize=(6, 4))
                fig.patch.set_facecolor("#1E2130")
                ax.set_facecolor("#1E2130")
                labels_short = [FEATURE_LABELS[f] for f in fi_df["feature"]][::-1]
                ax.barh(
                    labels_short,
                    fi_df["importance"].values[::-1],
                    color="#7B9FE0",
                    height=0.6,
                    edgecolor="#2E3250",
                )
                ax.set_xlabel("Mean |SHAP Value|", color="#8B9DC3", fontsize=9)
                ax.tick_params(colors="#8B9DC3", labelsize=8)
                for spine in ax.spines.values():
                    spine.set_edgecolor("#2E3250")
                plt.tight_layout()
                st.pyplot(fig)

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

**Example interpretation:** For , blue points (low volatility) appear on the right (positive SHAP = pushes toward Bot). This confirms that accounts with low behavioural variability are more likely to be classified as bots.
                """
            )
    else:
        st.info("Upload `shap_summary.png` to your repository.")

    # ── SHAP Feature-by-Feature Beeswarm Interpretation ──────────────────────
    st.markdown("---")
    st.subheader("🔎 SHAP: Feature-by-Feature Interpretation")
    st.markdown(
        "Each expandable card below explains what the beeswarm plot reveals about "
        "that specific feature's influence on the Bot vs. Organic prediction."
    )
    beeswarm_insights_ov = [
        ("behavioral_volatility_index",
         "🔵 Blue dots (low volatility) appear far to the **right** — strongly pushing toward Bot. "
         "🔴 Red dots (high volatility) appear to the **left** — pushing toward Organic. "
         "**Conclusion:** Low behavioural regularity is the clearest bot signature the model found. "
         "Bots operate like scheduled scripts — their activity patterns are highly uniform."),
        ("comment_similarity_score",
         "🔴 Red dots (high similarity) cluster to the **right** — pushing toward Bot. "
         "Bots post near-identical, copy-pasted comments repeatedly across posts. "
         "**Conclusion:** Templated, repetitive commenting is a strong and reliable automated activity marker."),
        ("follower_following_ratio",
         "🔴 Red dots (high ratio) are spread far to the **right** — a strong bot signal. "
         "Bots aggressively follow many accounts in bulk to gain followers back, inflating their ratio. "
         "**Conclusion:** An unusually high follower/following ratio is a reliable and independent bot indicator."),
        ("posting_frequency",
         "🔴 Red dots (high frequency) lean to the **right** — pushing toward Bot. "
         "Automated accounts post continuously around the clock to maximise reach. "
         "**Conclusion:** Abnormally high posting rates are a consistent signal of bot-driven automation."),
        ("timing_regularity_score",
         "Points cluster tightly near zero with a slight rightward lean for higher values. "
         "**Conclusion:** Some regularity signal exists but is largely absorbed by the volatility index — "
         "it provides partially redundant information and contributes less independently."),
        ("engagement_burst_ratio",
         "Most points hug zero, with a few outlier red dots to the right. "
         "**Conclusion:** Burst engagement is an occasional but not universal bot signal — "
         "it fires for coordinated bot networks but is absent in simpler automated accounts."),
        ("interaction_density_score",
         "Very tight clustering near zero for almost all samples. "
         "**Conclusion:** This feature contributes almost no unique information — "
         "its signal is entirely captured by posting frequency and comment similarity, "
         "which is why it ranks last in SHAP importance."),
    ]
    for feat_ov, insight_ov in beeswarm_insights_ov:
        with st.expander(f"**{FEATURE_LABELS[feat_ov]}**"):
            st.markdown(insight_ov)

    # ── Feature Importance Bar Chart + Ranked Insights ────────────────────────
    st.markdown("---")
    st.subheader("📊 SHAP Feature Importance")
    st.markdown(
        "Features are ranked by their **mean absolute SHAP value** — "
        "a measure of each feature's average contribution to the model's predictions across the entire test set."
    )
    if fi_df is not None:
        fig_fi2, ax_fi2 = plt.subplots(figsize=(7, 4))
        fig_fi2.patch.set_facecolor("#1E2130")
        ax_fi2.set_facecolor("#1E2130")
        labels_fi2 = [FEATURE_LABELS[f] for f in fi_df["feature"]][::-1]
        ax_fi2.barh(labels_fi2, fi_df["importance"].values[::-1],
                    color="#7B9FE0", height=0.6, edgecolor="#2E3250")
        ax_fi2.set_xlabel("Mean |SHAP Value|", color="#8B9DC3", fontsize=9)
        ax_fi2.tick_params(colors="#8B9DC3", labelsize=8)
        for spine in ax_fi2.spines.values():
            spine.set_edgecolor("#2E3250")
        plt.tight_layout()
        st.pyplot(fig_fi2)

        fi_display2 = fi_df.copy()
        fi_display2["Rank"]        = range(1, len(fi_df) + 1)
        fi_display2["Feature"]     = fi_df["feature"].map(FEATURE_LABELS)
        fi_display2["Mean |SHAP|"] = fi_df["importance"].round(4)
        fi_display2["Relative %"]  = (fi_df["importance"] / fi_df["importance"].sum() * 100).round(1)
        st.dataframe(
            fi_display2[["Rank", "Feature", "Mean |SHAP|", "Relative %"]],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### 🔎 What Each Feature Tells the Model")
        fi_insights = [
            ("🥇", "Behavioral Volatility Index", "0.1740", "35.0%",
             "The single strongest signal. **Bots behave predictably** — their activity is scripted and repetitive, resulting in very low variance. "
             "A low volatility score strongly pushes the model toward a Bot prediction. "
             "Human users have irregular, spontaneous behaviour that produces high volatility."),
            ("🥈", "Comment Similarity Score", "0.1378", "27.7%",
             "The second most important feature. **Bots reuse templated comments** — copy-pasted promotional text or near-identical replies — "
             "leading to high semantic similarity across their comment history. "
             "Organic users write varied, context-specific comments that keep this score low."),
            ("🥉", "Follower / Following Ratio", "0.0986", "19.8%",
             "Despite showing weak correlation with all other features, this ranks 3rd in SHAP importance. "
             "This confirms it captures an **independent dimension** of bot behaviour: bots aggressively follow many accounts "
             "to gain followers back, inflating their following count disproportionately."),
            ("4️⃣", "Posting Frequency", "0.0491", "9.9%",
             "Bots post far more frequently than organic users to maximise reach. "
             "A high posting rate combined with high comment similarity is a reliable compound signal of automated activity."),
            ("5️⃣", "Timing Regularity Score", "0.0315", "6.3%",
             "Automated accounts post at highly regular intervals — like a scheduled task. "
             "While useful, this feature overlaps with behavioral volatility, reducing its individual SHAP contribution."),
            ("6️⃣", "Engagement Burst Ratio", "0.0273", "5.5%",
             "Coordinated bot networks produce sudden, simultaneous engagement spikes on specific posts. "
             "This feature captures those bursts, though it is partially redundant with timing regularity."),
            ("7️⃣", "Interaction Density Score", "0.0166", "3.3%",
             "The weakest predictor. Its information is largely **already captured** by posting frequency and comment similarity, "
             "as confirmed by their high inter-correlations (r > 0.6) in the correlation analysis."),
        ]
        for rank, feat_name, shap_val, pct, desc in fi_insights:
            with st.expander(f"{rank}  **{feat_name}** — Mean |SHAP| `{shap_val}` · Relative contribution `{pct}`"):
                st.markdown(desc)


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

            st.pyplot(shap_bar_fig(shap_1d, title="Feature Contributions to Bot Prediction"))
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
# PAGE 4 — MODEL INSIGHTS (Dataset Explorer only)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Insights":
    st.title("📈 Dataset Explorer")
    st.markdown(
        "Explore the distribution of each behavioural feature across Bot and Organic accounts. "
        "Select a feature from the dropdown to view how the two classes differ."
    )

    _csv_path = _find_file("social_media_dataset.csv")
    if _csv_path:
        df_data = pd.read_csv(_csv_path)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records",    len(df_data))
        c2.metric("Bot Accounts",     int((df_data["label"] == 1).sum()))
        c3.metric("Organic Accounts", int((df_data["label"] == 0).sum()))

        st.markdown("---")
        st.markdown("**Feature Distributions by Class**")
        feat_sel = st.selectbox(
            "Select feature to visualise",
            FEATURE_COLS,
            format_func=lambda x: FEATURE_LABELS[x],
        )
        fig, ax = plt.subplots(figsize=(8, 3.5))
        fig.patch.set_facecolor("#1E2130")
        ax.set_facecolor("#1E2130")
        for label_val, color, name in [(0, "#21C55D", "Organic"), (1, "#FF4B4B", "Bot")]:
            subset = df_data[df_data["label"] == label_val][feat_sel]
            ax.hist(subset, bins=40, alpha=0.65, color=color, label=name, edgecolor="none")
        ax.set_xlabel(FEATURE_LABELS[feat_sel], color="#8B9DC3", fontsize=9)
        ax.set_ylabel("Count", color="#8B9DC3", fontsize=9)
        ax.set_title(f"Distribution of {FEATURE_LABELS[feat_sel]}", color="#A8C0E8",
                     fontsize=11, fontweight="bold")
        ax.tick_params(colors="#8B9DC3")
        ax.legend(facecolor="#252A3D", edgecolor="#2E3250", labelcolor="#E0E0E0")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2E3250")
        plt.tight_layout()
        st.pyplot(fig)

        dist_insights = {
            "timing_regularity_score":
                "**What the distribution reveals:** Bot accounts (red) are shifted toward higher values "
                "with a tighter spread, confirming they post at more predictable intervals. "
                "Organic users (green) show a broader, lower-centred distribution reflecting natural variation.",
            "engagement_burst_ratio":
                "**What the distribution reveals:** Both classes overlap heavily at lower values, "
                "but bots have a longer right tail — occasional extreme burst events that organic users rarely produce.",
            "comment_similarity_score":
                "**What the distribution reveals:** The two distributions are clearly separated. "
                "Bots cluster at higher similarity scores (templated comments), while organic users "
                "spread across lower values (varied, unique comments).",
            "interaction_density_score":
                "**What the distribution reveals:** The distributions largely overlap, which is consistent "
                "with this feature being the weakest SHAP contributor — it does not cleanly separate bots from organic users.",
            "follower_following_ratio":
                "**What the distribution reveals:** Organic users (green) concentrate near a ratio of 1 "
                "(balanced following/followers). Bots (red) extend further right with higher ratios, "
                "reflecting their aggressive following strategy.",
            "posting_frequency":
                "**What the distribution reveals:** Organic users post infrequently (left-clustered). "
                "Bots show a wider spread at higher frequencies, consistent with automated, high-volume posting.",
            "behavioral_volatility_index":
                "**What the distribution reveals:** This is the cleanest separation of all seven features. "
                "Bots cluster at very low volatility values (scripted, predictable). "
                "Organic users spread across higher values (irregular, human behaviour). "
                "This directly explains why it is the #1 SHAP feature.",
        }
        st.info(dist_insights.get(feat_sel, ""))

        with st.expander("Preview raw data"):
            st.dataframe(df_data.head(50), use_container_width=True)
    else:
        st.info(
            "Dataset not found. Run `generate_dataset.py` to create "
            "`social_media_dataset.csv` in the project root."
        )
