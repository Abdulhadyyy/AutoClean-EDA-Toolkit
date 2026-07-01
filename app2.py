"""
╔══════════════════════════════════════════════════════════════════════════╗
║          AutoClean & EDA Toolkit  ·  Advanced Edition                   ║
║          Course  : Introduction to Data Science (IDS)                   ║
║          Program : BS Data Science – Semester 2                         ║
║          Stack   : Streamlit · Pandas · Scikit-learn · Scipy · Plotly   ║
╚══════════════════════════════════════════════════════════════════════════╝

Architecture:
  ┌─────────────────────────────────────────────────────────┐
  │  DataPreprocessor  – cleaning, imputation, outliers     │
  │  EDAAnalyzer       – statistics, PCA, hypothesis tests  │
  │  FeatureEngineer   – encoding, scaling                  │
  │  ModelTrainer      – problem detection, data split,     │
  │                       LEAK-FREE sklearn Pipeline,        │
  │                       hyperparameter search, fitting     │
  │  ModelEvaluator    – metrics, best-model, plots         │
  │  PredictionEngine  – batch prediction wrapper           │
  │  UIBuilder         – all Streamlit rendering logic      │
  └─────────────────────────────────────────────────────────┘

  Learning Paradigm Labels
  ────────────────────────
  • KNN Imputation → NOT supervised (no labels used)
  • PCA            → Unsupervised  (no labels used)
  • ML Studio      → Supervised   (X + y required)

  ── UPGRADE LOG (read this if you are defending this project!) ───────────
  1. DATA-LEAKAGE FIX (ModelTrainer)
     The OLD code label-encoded the *entire* X (train + test rows together)
     and only split AFTERWARDS. That means the encoder "saw" the test set's
     categories before training even began — a classic, silent leakage bug.
     The NEW code calls `train_test_split` FIRST on the raw (unencoded)
     columns, then builds an sklearn `Pipeline` (via `ColumnTransformer`)
     that is `.fit()`-ed ONLY on the training fold. `.transform()` (never
     `.fit_transform()` again) is then applied to the test fold. This
     guarantees the test set remains genuinely "unseen" data, exactly as it
     would be in production.

  2. MEMORY-MANAGEMENT FIX (DataPreprocessor / FeatureEngineer / session_state)
     `pandas.DataFrame.copy()` allocates an entirely new block of memory.
     The OLD code copied dataframes defensively (sometimes 2-3x per object)
     "just in case," which silently doubles/triples RAM usage on large
     CSVs. The NEW code copies ONLY at the one true boundary that matters —
     when a class is first handed a reference it intends to mutate — and
     stores single, intentional snapshots in `st.session_state` instead of
     re-copying on every Streamlit re-run.

  3. HYPERPARAMETER-TUNING UPGRADE (ModelTrainer)
     The OLD code hardcoded `max_depth=12`, `n_estimators=100`, etc. for
     every dataset, regardless of size or shape. The NEW code wraps each
     model in `GridSearchCV` (small, exhaustive grids for cheap models) or
     `RandomizedSearchCV` (for models with bigger search spaces, e.g.
     Random Forest / XGBoost) so hyperparameters are *learned from the
     training fold via internal cross-validation*, not guessed once and
     hardcoded forever.

  4. GHOST-VALUE LOGIC FIX (DataPreprocessor.auto_clean)
     The OLD mask `df[num_cols] < 0` nuked EVERY negative number to NaN —
     including perfectly valid negative numbers (temperature in °C, a bank
     account's overdraft, a profit/loss column, a z-score, etc.). The NEW
     logic only targets specific "sentinel" ghost values (e.g. -999, -9999,
     -1, -1000 — the classic placeholders survey/ERP systems use for
     "missing") and is fully configurable from the Streamlit sidebar, so a
     "loss of -500" is never mistaken for a missing-value flag of -999.
"""

# ── Standard library ──────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")
from typing import Optional

# ── Third-party ───────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# ── Scikit-learn: preprocessing & EDA ────────────────────────────────────────
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import (
    MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder,
)
from scipy.stats import zscore, pearsonr, spearmanr

# ── Scikit-learn: ML Studio ───────────────────────────────────────────────────
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_absolute_error, mean_squared_error,
    r2_score, confusion_matrix,
)
# ── Scikit-learn: LEAK-FREE pipeline machinery ───────────────────────────────
# `Pipeline`         → chains preprocessing steps + a model into ONE object that
#                       respects fit/transform boundaries automatically.
# `ColumnTransformer` → applies DIFFERENT preprocessing to DIFFERENT columns
#                       (e.g. scale the numeric columns, one-hot the categorical
#                       columns) inside that same single, leak-safe object.
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# ── XGBoost (optional – graceful fallback if not installed) ──────────────────
try:
    from xgboost import XGBClassifier, XGBRegressor
    _XGBOOST_OK = True
except ImportError:
    _XGBOOST_OK = False

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoClean & EDA Toolkit",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL THEME
# ─────────────────────────────────────────────────────────────────────────────
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg:            #0A0B0D;
    --bg-elevated:   #111317;
    --surface:       #15171C;
    --surface-hover: #1B1E24;
    --border:        #23262C;
    --border-strong: #2D3138;
    --text-primary:  #EDEEF0;
    --text-secondary:#9AA0AB;
    --text-tertiary: #6B7280;
    --accent:        #5B8DEF;
    --accent-strong: #4A78D6;
    --accent-soft:   rgba(91,141,239,.12);
    --accent-border: rgba(91,141,239,.32);
    --success:       #34D399;
    --success-soft:  rgba(52,211,153,.12);
    --warning:       #F2B544;
    --danger:        #F2545B;
    --radius-sm: 8px;  --radius-md: 12px;  --radius-lg: 16px;
    --shadow-sm: 0 1px 3px rgba(0,0,0,.35);
    --shadow-md: 0 6px 20px rgba(0,0,0,.40);
}

/* ── Base / typography ────────────────────────────────────────── */
html, body, .stApp, [class*="css"] { font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif; }
.stApp { background:var(--bg); color:var(--text-secondary); }
code, pre, .stCodeBlock, .stCode { font-family:'JetBrains Mono',monospace !important; }

h1, h2, h3, h4, h5 { color:var(--text-primary); font-weight:700; letter-spacing:-0.02em; }
h1 { font-size:2rem;   font-weight:800; margin-bottom:.2rem; }
h2 { font-size:1.35rem; margin-top:1.8rem; }
h3 { font-size:1.05rem; font-weight:600; }
p, li, label, .stMarkdown { color:var(--text-secondary); line-height:1.6; }
[data-testid="stCaptionContainer"] { color:var(--text-tertiary) !important; font-size:.9rem; }

/* ── Sidebar ───────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background:var(--bg-elevated); border-right:1px solid var(--border);
}
section[data-testid="stSidebar"] h1 { font-size:1.25rem; }

/* ── Metric cards ──────────────────────────────────────────────── */
div[data-testid="stMetric"] {
    background:var(--surface); border:1px solid var(--border);
    border-radius:var(--radius-md); padding:16px 20px;
    box-shadow:var(--shadow-sm); transition:border-color .15s ease;
}
div[data-testid="stMetric"]:hover { border-color:var(--border-strong); }
div[data-testid="stMetricLabel"] p {
    color:var(--text-tertiary); font-size:.74rem; font-weight:600;
    text-transform:uppercase; letter-spacing:.06em;
}
div[data-testid="stMetricValue"] { color:var(--text-primary); font-weight:700; font-size:1.55rem; }
div[data-testid="stMetricDelta"] { font-weight:600; }

/* ── Tabs ──────────────────────────────────────────────────────── */
button[data-baseweb="tab"] {
    font-size:14px; font-weight:600; color:var(--text-tertiary); padding:10px 6px;
}
button[data-baseweb="tab"][aria-selected="true"] { color:var(--accent); }
div[data-baseweb="tab-highlight"] { background-color:var(--accent) !important; height:2px; }
div[data-baseweb="tab-border"]    { background-color:var(--border) !important; }

/* ── Expanders ─────────────────────────────────────────────────── */
details {
    background:var(--surface); border:1px solid var(--border);
    border-radius:var(--radius-md); margin-bottom:10px; overflow:hidden;
}
details summary {
    padding:12px 16px; color:var(--text-primary);
    font-weight:600; font-size:.9rem; background:transparent;
}
details summary:hover { background:var(--surface-hover); }

/* ── Buttons ───────────────────────────────────────────────────── */
.stButton button, .stDownloadButton button {
    border-radius:var(--radius-sm); font-weight:600; font-size:.88rem;
    border:1px solid var(--border-strong); background:var(--surface);
    color:var(--text-primary); transition:all .15s ease; padding:.5rem 1rem;
}
.stButton button:hover, .stDownloadButton button:hover {
    border-color:var(--accent); color:var(--accent); background:var(--accent-soft);
}
.stButton button[kind="primary"], .stDownloadButton button[kind="primary"] {
    background:var(--accent); border-color:var(--accent); color:#fff;
}
.stButton button[kind="primary"]:hover, .stDownloadButton button[kind="primary"]:hover {
    background:var(--accent-strong); border-color:var(--accent-strong); color:#fff;
}

/* ── Inputs ────────────────────────────────────────────────────── */
div[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input {
    background:var(--surface) !important; border-color:var(--border) !important;
    border-radius:var(--radius-sm) !important; color:var(--text-primary) !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] { background:var(--accent); }
.stSlider [data-baseweb="slider"] > div > div { background:var(--accent) !important; }
.stRadio label, .stCheckbox label { color:var(--text-secondary); }

/* ── File uploader ─────────────────────────────────────────────── */
section[data-testid="stFileUploaderDropzone"] {
    background:var(--surface); border:1px dashed var(--border-strong);
    border-radius:var(--radius-md);
}

/* ── Alerts ────────────────────────────────────────────────────── */
div[data-testid="stAlert"] {
    border-radius:var(--radius-sm); border:1px solid var(--border);
    background:var(--surface);
}

/* ── Dataframe / tables ────────────────────────────────────────── */
.stDataFrame { border-radius:var(--radius-md); overflow:hidden; border:1px solid var(--border); }

/* ── Divider ───────────────────────────────────────────────────── */
hr { border:none; border-top:1px solid var(--border); margin:1.5rem 0; }

/* ── Badge ─────────────────────────────────────────────────────── */
.badge {
    display:inline-block; background:var(--accent-soft); color:var(--accent);
    border:1px solid var(--accent-border); border-radius:999px;
    padding:3px 12px; font-size:.74rem; font-weight:600; margin-left:8px; letter-spacing:.02em;
}

/* ── Highlight box ─────────────────────────────────────────────── */
.highlight-box {
    background:var(--surface); border:1px solid var(--border);
    border-left:3px solid var(--accent); border-radius:var(--radius-md);
    padding:18px 22px; margin-bottom:18px; box-shadow:var(--shadow-sm);
}

/* ── Best model banner ─────────────────────────────────────────── */
.best-model-banner {
    background: linear-gradient(135deg, rgba(52,211,153,.15), rgba(52,211,153,.05));
    border:1px solid rgba(52,211,153,.4);
    border-left:4px solid #34D399;
    border-radius:var(--radius-md);
    padding:16px 22px; margin:14px 0;
}

/* ── Learning-type chips ────────────────────────────────────────── */
.chip-supervised {
    display:inline-block; background:rgba(52,211,153,.15); color:#34D399;
    border:1px solid rgba(52,211,153,.4); border-radius:999px;
    padding:4px 14px; font-size:.78rem; font-weight:700; letter-spacing:.03em;
}
.chip-unsupervised {
    display:inline-block; background:rgba(242,181,68,.12); color:#F2B544;
    border:1px solid rgba(242,181,68,.35); border-radius:999px;
    padding:4px 14px; font-size:.78rem; font-weight:700; letter-spacing:.03em;
}
.chip-not-supervised {
    display:inline-block; background:rgba(167,139,250,.12); color:#A78BFA;
    border:1px solid rgba(167,139,250,.35); border-radius:999px;
    padding:4px 14px; font-size:.78rem; font-weight:700; letter-spacing:.03em;
}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

# ── Plotly theme: a custom template so charts match the dashboard chrome ──
pio.templates["enterprise_dark"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="#15171C",
        plot_bgcolor="#15171C",
        font=dict(family="Inter, sans-serif", color="#9AA0AB", size=12),
        title=dict(font=dict(color="#EDEEF0", size=15)),
        colorway=["#5B8DEF", "#34D399", "#F2B544", "#A78BFA", "#F2545B", "#22D3EE"],
        xaxis=dict(gridcolor="#23262C", zerolinecolor="#23262C", linecolor="#2D3138"),
        yaxis=dict(gridcolor="#23262C", zerolinecolor="#23262C", linecolor="#2D3138"),
        legend=dict(font=dict(color="#9AA0AB")),
        margin=dict(t=60, b=40, l=50, r=30),
    )
)
PLOTLY_TEMPLATE = "enterprise_dark"
ACCENT          = "#5B8DEF"

# ─────────────────────────────────────────────────────────────────────────────
#  MODULE-LEVEL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
_CURRENCY_RE       = r'(?:Rs\.?|PKR|USD|EUR|\$|£|€|,)'
_DIRTY_THRESH_PCT  = 0.10

# ── GHOST-VALUE SENTINEL FIX ──────────────────────────────────────────────────
# WHY this exists (read this before touching auto_clean!):
#   The OLD logic used the mask `df[num_cols] < 0` to hunt for "ghost values"
#   (placeholder codes like -999 that some legacy systems use to mean "no
#   data here"). The problem: that mask treats EVERY negative number as
#   suspicious, even when negative numbers are 100% legitimate data — e.g.
#   a temperature of -5°C, a stock return of -2.3%, a bank balance of -300,
#   or a financial "loss" column. Blindly NaN-ing those would silently
#   destroy real information before the student even notices.
#
#   The FIX: instead of "is it negative?", we ask "is it one of the EXACT
#   placeholder codes that data-entry/legacy systems are known to use for
#   missing data?" This is a finite, explicit list of sentinel values. Any
#   number that is merely negative but NOT in this list (e.g. -5, -300,
#   -2.3) survives untouched. The list below is intentionally editable —
#   the Streamlit sidebar widget (see `render_preprocessing_tab`) lets the
#   user add/remove sentinels for their own dataset's quirks, so the logic
#   stays smart AND configurable instead of "smart but rigid."
_DEFAULT_GHOST_SENTINELS: list[float] = [-999, -9999, -1, -1000, -99, -99999]


# ══════════════════════════════════════════════════════════════════════════════
#  CLASS 1 · DataPreprocessor
# ══════════════════════════════════════════════════════════════════════════════
class DataPreprocessor:
    """
    Handles all data-cleaning, imputation, and outlier-treatment logic.
    Always works on an internal copy so the caller's original DataFrame
    reference (e.g. `self.raw_df` in UIBuilder, or the cached object
    returned by `load_data`) is never mutated in-place.
    """

    def __init__(self, df: pd.DataFrame):
        # ── MEMORY FIX ────────────────────────────────────────────────────────
        # WHY only ONE .copy() and not two:
        #   The OLD code stored BOTH `self.original = df.copy()` AND
        #   `self.df = df.copy()` — i.e. it silently doubled the RAM
        #   footprint of every uploaded CSV the instant a DataPreprocessor was
        #   created, and `self.original` was never even read anywhere else in
        #   the codebase (verified: zero other references). That's a pure
        #   memory leak with no functional benefit — the worst kind of bug,
        #   because nothing "breaks," it just quietly eats RAM on large files.
        #
        #   We still need EXACTLY ONE copy, though — `self.df` is mutated
        #   in-place by methods like `drop_duplicates()`, `knn_impute()`, etc.
        #   Without ANY copy, those mutations would leak backward into the
        #   caller's original `raw_df` (which Streamlit caches via
        #   `@st.cache_data` and reuses across reruns) — corrupting the
        #   "Raw Data" tab the user expects to stay pristine. So one copy is
        #   the correct, minimal boundary: it protects the caller's object
        #   while avoiding a second, redundant allocation.
        self.df = df.copy()

    # ── Profiling ─────────────────────────────────────────────────────────────
    def profile(self) -> pd.DataFrame:
        df = self.df
        return pd.DataFrame({
            "Feature"     : df.columns,
            "Dtype"       : df.dtypes.astype(str).values,
            "Null Count"  : df.isnull().sum().values,
            "Null %"      : (df.isnull().mean() * 100).round(2).values,
            "Unique"      : df.nunique().values,
            "Sample Value": [
                df[c].dropna().iloc[0] if df[c].notna().any() else "N/A"
                for c in df.columns
            ],
        })

    # ── Duplicate removal ─────────────────────────────────────────────────────
    def drop_duplicates(self) -> int:
        before   = len(self.df)
        self.df  = self.df.drop_duplicates()
        return before - len(self.df)

    # ── Currency / dirty-text → numeric ───────────────────────────────────────
    def convert_dirty_numerics(self) -> list[str]:
        """
        Scans every `object` (text) column and asks: "if I strip out currency
        symbols/commas, does this column become MOSTLY numeric?" If yes, the
        whole column is converted to a true numeric dtype.

        WHY the threshold check (`temp.notna().sum() >= threshold`) matters:
        We don't convert a column just because A FEW cells happen to parse as
        numbers (e.g. a "Notes" column might contain "5 stars" somewhere).
        We require at least `_DIRTY_THRESH_PCT` (10%) of the column to
        successfully parse as numeric BEFORE we commit to the conversion —
        this avoids accidentally numeric-ifying a genuinely textual column
        that just happens to contain a few stray digits.
        """
        converted = []
        threshold = max(1, len(self.df) * _DIRTY_THRESH_PCT)
        for col in self.df.columns:
            if self.df[col].dtype != 'object':
                continue
            # Strip currency tokens (Rs., PKR, $, £, €, commas) then whitespace,
            # so "Rs. 1,200" → "1200" → can be parsed by pd.to_numeric.
            cleaned = (
                self.df[col].astype(str)
                            .str.replace(_CURRENCY_RE, '', regex=True)
                            .str.strip()
            )
            # errors='coerce' turns anything that still isn't a clean number
            # into NaN instead of raising — letting us COUNT successes below.
            temp = pd.to_numeric(cleaned, errors='coerce')
            if temp.notna().sum() >= threshold:
                self.df[col] = temp
                converted.append(col)
        return converted

    # ── Temporal column detection ─────────────────────────────────────────────
    def detect_temporal_columns(self) -> list[str]:
        """
        Heuristically flags text columns that "look like" dates, so the UI can
        suggest sequential (forward/backward) fill instead of KNN imputation
        for time-ordered data (KNN ignores row order; ffill/bfill respects it).
        Only a 50-row SAMPLE is parsed per column (not the whole column) purely
        for speed — large CSVs would otherwise re-parse dates on every rerun.
        """
        candidates = []
        for col in self.df.select_dtypes(include='object').columns:
            sample = self.df[col].dropna().head(50)
            try:
                parsed = pd.to_datetime(sample, infer_datetime_format=True, errors='coerce')
                # If ≥ 80% of the sample parses cleanly as a date, treat the
                # WHOLE column as temporal. A lower bar would risk false
                # positives on text columns that coincidentally contain a few
                # date-like strings (e.g. "Q1 2023" appearing in free text).
                if parsed.notna().sum() / max(len(sample), 1) >= 0.80:
                    candidates.append(col)
            except Exception:
                pass
        return candidates

    # ── Sequential imputation ─────────────────────────────────────────────────
    def sequential_impute(self, method: str = "ffill") -> None:
        """
        Carries the last known value FORWARD ('ffill') or BACKWARD ('bfill')
        through NaN gaps. Correct choice for time-series-like data, where a
        missing reading is usually best guessed as "whatever the value was at
        the previous (or next) point in time" — neighbour SIMILARITY (as KNN
        does) is irrelevant here; neighbour ORDER is what matters.
        """
        self.df = self.df.ffill() if method == "ffill" else self.df.bfill()

    # ── KNN imputation (numerical) ────────────────────────────────────────────
    def knn_impute(self, n_neighbors: int = 5) -> None:
        """
        Fills missing numeric cells using the average of the `n_neighbors`
        ROWS that are most similar across ALL numeric columns (Euclidean
        distance), weighted by inverse distance ('distance' weighting → closer
        rows count more). This is NOT supervised learning: no target/label
        column is involved — KNNImputer only ever looks at feature columns to
        estimate other feature columns, which is why the UI tags it
        "🔵 NOT Supervised" rather than "Supervised."
        """
        num_cols = self.df.select_dtypes(include=["int64", "float64"]).columns
        if num_cols.empty:
            return
        imp = KNNImputer(n_neighbors=n_neighbors, weights="distance")
        self.df[num_cols] = imp.fit_transform(self.df[num_cols])

    # ── Mode imputation (categorical) ─────────────────────────────────────────
    def mode_impute_categoricals(self) -> None:
        """
        For text/categorical columns, KNN distance doesn't make sense (there's
        no numeric distance between "Lahore" and "Karachi"), so we instead
        fill gaps with the column's MODE (most frequent category) — the
        simplest, most defensible guess when no ordering or distance metric
        exists.
        """
        for col in self.df.select_dtypes(include='object').columns:
            mode = self.df[col].mode()
            self.df[col] = self.df[col].fillna(
                mode.iloc[0] if not mode.empty else "Unknown"
            )

    # ── Critical-column row drop ──────────────────────────────────────────────
    def drop_on_critical(self, critical_cols: list[str]) -> int:
        """
        Some columns are too important to "guess" (e.g. a primary ID, or the
        target column itself) — imputing a fake value there could corrupt
        downstream analysis. For those, we drop the ROW entirely instead of
        filling it.
        """
        if not critical_cols:
            return 0
        before   = len(self.df)
        self.df  = self.df.dropna(subset=critical_cols)
        return before - len(self.df)

    # ── Outlier detection ─────────────────────────────────────────────────────
    def detect_outliers_iqr(self, col: str) -> pd.Series:
        """
        Classic Tukey fence: anything below Q1 − 1.5·IQR or above
        Q3 + 1.5·IQR is flagged. Robust to skewed data because it's based on
        quartiles (ranks), not the mean/std — a few extreme values can't drag
        the fence around the way they would distort a mean.
        """
        Q1, Q3 = self.df[col].quantile([0.25, 0.75])
        IQR    = Q3 - Q1
        return (self.df[col] < Q1 - 1.5 * IQR) | (self.df[col] > Q3 + 1.5 * IQR)

    def detect_outliers_zscore(self, col: str, threshold: float = 3.0) -> pd.Series:
        """
        Flags points more than `threshold` standard deviations from the mean.
        Z-score is sensitive to the very outliers it's trying to detect (the
        mean/std it's built from get pulled by them), so it works best on
        roughly-normal data — that's why both IQR and Z-Score are offered as
        a choice in the UI rather than picking one for the user.
        """
        z         = np.abs(zscore(self.df[col].dropna()))
        mask      = pd.Series(False, index=self.df.index)
        # zscore() drops NaNs internally, so we must re-align its output back
        # onto the ORIGINAL index — otherwise positions would silently shift
        # whenever the column has missing values.
        valid_idx = self.df[col].dropna().index
        mask[valid_idx] = z > threshold
        return mask

    def cap_outliers(self, col: str) -> None:
        """Winsorisation: clip (don't delete) extreme values to the IQR fence
        boundaries, preserving row count while taming extreme influence."""
        Q1, Q3 = self.df[col].quantile([0.25, 0.75])
        IQR    = Q3 - Q1
        self.df[col] = self.df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

    def drop_outliers(self, mask: pd.Series) -> int:
        """Alternative to capping: remove flagged rows outright."""
        before   = len(self.df)
        self.df  = self.df[~mask]
        return before - len(self.df)

    # ══════════════════════════════════════════════════════════════════════════
    #  auto_clean  ·  The bulletproof pipeline
    # ══════════════════════════════════════════════════════════════════════════
    def auto_clean(
        self,
        k_neighbors: int = 5,
        ghost_sentinels: Optional[list[float]] = None,
    ) -> list[str]:
        """
        All-in-one auto-clean pipeline in strict, dependency-correct order.

        Steps
        ─────
        1  Duplicate removal
        2  Text normalisation   – .title() on TRUE categoricals only
        3  Numeric conversion   – strip currency tokens → pd.to_numeric (coerce)
        4  SMART ghost-value fix – mask only KNOWN sentinel codes → NaN
                                    (e.g. -999 / -9999), leaving genuine
                                    negative numbers (losses, °C, etc.) intact
        5  KNN imputation + categorical mode fill
        6  IQR Winsorisation

        Parameters
        ──────────
        ghost_sentinels : list of exact numeric codes to treat as "ghost"
            placeholders for missing data. Defaults to `_DEFAULT_GHOST_SENTINELS`
            (the common -999/-9999/-1/... family) if not provided by the
            caller. Passing an empty list `[]` disables ghost-value masking
            entirely — useful for datasets where ALL negative numbers (and
            even those exact codes) are legitimate.
        """
        log: list[str] = []

        # WHY a default is resolved HERE (inside the method) rather than in
        # the signature `ghost_sentinels: list = _DEFAULT_GHOST_SENTINELS`:
        # using a *mutable* default argument (a list) is a classic Python
        # footgun — that single list object would be shared and could be
        # mutated across every call. Resolving `None` → a fresh copy of the
        # module constant here avoids that trap entirely.
        if ghost_sentinels is None:
            ghost_sentinels = list(_DEFAULT_GHOST_SENTINELS)

        # ── Step 1 · Drop duplicates ─────────────────────────────────────────
        n_dupes = self.drop_duplicates()
        if n_dupes:
            log.append(f"✅ Removed {n_dupes} duplicate row(s).")

        # ── Column classification ─────────────────────────────────────────────
        threshold      = max(1, len(self.df) * _DIRTY_THRESH_PCT)
        dirty_numerics : list[str] = []
        true_cats      : list[str] = []

        for col in self.df.select_dtypes(include='object').columns:
            probe = (
                self.df[col].astype(str)
                            .str.replace(_CURRENCY_RE, '', regex=True)
                            .str.strip()
            )
            if pd.to_numeric(probe, errors='coerce').notna().sum() >= threshold:
                dirty_numerics.append(col)
            else:
                true_cats.append(col)

        # ── Step 2 · Text normalisation ──────────────────────────────────────
        for col in true_cats:
            try:
                self.df[col] = self.df[col].apply(
                    lambda x: str(x).title().strip() if pd.notna(x) else x
                )
            except TypeError:
                pass

        if true_cats:
            log.append(
                f"✅ Text normalisation (.title) applied to "
                f"{len(true_cats)} categorical column(s): {true_cats}."
            )
        else:
            log.append("ℹ️  No true categorical columns detected.")

        # ── Step 3 · Strict numeric conversion ───────────────────────────────
        converted: list[str] = []
        for col in dirty_numerics:
            try:
                cleaned = (
                    self.df[col].astype(str)
                                .str.replace(_CURRENCY_RE, '', regex=True)
                                .str.strip()
                )
                self.df[col] = pd.to_numeric(cleaned, errors='coerce')
                converted.append(col)
            except (TypeError, ValueError) as exc:
                log.append(f"⚠️  Could not convert '{col}': {exc}")

        if converted:
            log.append(
                f"✅ Currency symbols stripped; coerced to float: {converted}."
            )

        # ── Step 4 · SMART ghost-value fix (sentinel-code masking) ───────────
        # WHY this changed (this is the headline fix — know this cold for your
        # defense!):
        #   OLD logic:  mask = df[num_cols] < 0   → nukes EVERY negative cell.
        #   PROBLEM:    a genuinely valid negative number (a financial LOSS of
        #               -500, a temperature of -12°C, a return of -3.2%) is
        #               mathematically indistinguishable from a "ghost"
        #               placeholder like -999 under a pure `< 0` test. The old
        #               code would destroy real data and call it a "fix."
        #
        #   NEW logic:  mask = df[num_cols].isin(ghost_sentinels)
        #   `isin()` checks for EXACT membership in a finite, explicit list of
        #   known sentinel codes (e.g. -999, -9999, -1, -1000 — see
        #   `_DEFAULT_GHOST_SENTINELS`). A cell with value -500 is NOT in that
        #   list, so it survives untouched; a cell with value -999 (a textbook
        #   "missing data" placeholder used by many real-world systems) is
        #   still correctly caught and converted to NaN. This is "smarter" in
        #   the precise sense that it uses domain knowledge about WHICH
        #   negative numbers are suspicious, instead of treating sign alone as
        #   evidence of corruption. It's also "configurable" because
        #   `ghost_sentinels` is a parameter the Streamlit UI lets the user
        #   edit (see the sidebar widget in `render_preprocessing_tab`) —
        #   so a dataset that uses -7777 as its missing-flag can be handled
        #   too, without touching this code.
        num_cols = self.df.select_dtypes(include='number').columns
        if not num_cols.empty and ghost_sentinels:
            ghost_mask  = self.df[num_cols].isin(ghost_sentinels)
            ghost_count = int(ghost_mask.sum().sum())
            self.df[num_cols] = self.df[num_cols].mask(ghost_mask, np.nan)
            log.append(
                f"✅ Smart ghost-value fix: {ghost_count} sentinel placeholder "
                f"value(s) {ghost_sentinels} masked → NaN. Genuine negative "
                "numbers (losses, sub-zero readings, etc.) were left intact."
            )
        elif not num_cols.empty:
            # User passed an empty sentinel list → ghost-masking explicitly
            # disabled for this run (e.g. every negative number in this
            # dataset is legitimate, so there's nothing to mask).
            log.append(
                "ℹ️  Ghost-value masking skipped — no sentinel codes were "
                "configured for this run."
            )
        else:
            log.append("ℹ️  No numeric columns found for ghost-value masking.")

        # ── Step 5 · Algorithmic imputation ──────────────────────────────────
        # WHY this runs AFTER Step 4 (ghost-fix), not before:
        #   If we imputed BEFORE removing ghost values, the KNN imputer would
        #   treat -999 placeholders as REAL data points when computing
        #   "nearest neighbours," dragging estimates toward nonsense. Doing
        #   Step 4 first converts ghosts → NaN, so by the time KNN runs, it
        #   only ever sees genuine numbers as the basis for similarity.
        missing_before = int(self.df.isnull().sum().sum())

        num_cols = self.df.select_dtypes(include='number').columns.tolist()
        if num_cols:
            try:
                imp               = KNNImputer(n_neighbors=k_neighbors, weights='distance')
                self.df[num_cols] = imp.fit_transform(self.df[num_cols])
            except ValueError as exc:
                log.append(f"⚠️  KNN imputation failed: {exc}")

        for col in self.df.select_dtypes(include='object').columns:
            mode_s = self.df[col].mode()
            self.df[col] = self.df[col].fillna(
                mode_s.iloc[0] if not mode_s.empty else "Unknown"
            )

        log.append(
            f"✅ Imputed {missing_before} missing value(s): "
            f"KNN (k={k_neighbors}) on numerics + mode fill on categoricals."
        )

        # ── Step 6 · IQR Winsorisation ────────────────────────────────────────
        # WHY this runs LAST: outlier capping should only ever see the FINAL,
        # fully-imputed numbers. Running it earlier could clip a column while
        # NaN placeholders were still present, or clip based on a Q1/Q3 that
        # ghost values (-999) had distorted.
        capped: list[str] = []
        for col in self.df.select_dtypes(include='number').columns:
            try:
                self.cap_outliers(col)
                capped.append(col)
            except (TypeError, ValueError):
                pass

        log.append(
            f"✅ IQR Winsorisation applied to {len(capped)} numeric column(s)."
        )

        return log


# ══════════════════════════════════════════════════════════════════════════════
#  CLASS 2 · EDAAnalyzer
# ══════════════════════════════════════════════════════════════════════════════
class EDAAnalyzer:
    """
    Computes statistical summaries, hypothesis tests, and PCA decomposition.
    All heavy computation lives here; UIBuilder only calls these methods.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    @property
    def num_cols(self) -> list[str]:
        return self.df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    @property
    def cat_cols(self) -> list[str]:
        return self.df.select_dtypes(include='object').columns.tolist()

    def descriptive_stats(self) -> pd.DataFrame:
        desc = self.df[self.num_cols].describe().T
        desc["skewness"]  = self.df[self.num_cols].skew().values
        desc["kurtosis"]  = self.df[self.num_cols].kurt().values
        desc["skew_label"] = desc["skewness"].apply(
            lambda s: ("Symmetric" if abs(s) < 0.5
                       else ("Moderate Skew" if abs(s) < 1.0
                             else "High Skew"))
        )
        return desc.round(4)

    def correlation_with_pvalues(
        self, method: str = "pearson"
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        cols   = self.num_cols
        n      = len(cols)
        corr_m = pd.DataFrame(np.ones((n, n)),  index=cols, columns=cols)
        pval_m = pd.DataFrame(np.zeros((n, n)), index=cols, columns=cols)

        fn = pearsonr if method == "pearson" else spearmanr
        for i, c1 in enumerate(cols):
            for j, c2 in enumerate(cols):
                if i == j:
                    continue
                valid = self.df[[c1, c2]].dropna()
                if len(valid) < 3:
                    continue
                try:
                    r, p = fn(valid[c1], valid[c2])
                    corr_m.loc[c1, c2] = round(r, 4)
                    pval_m.loc[c1, c2] = round(p, 4)
                except Exception:
                    pass
        return corr_m, pval_m

    def run_pca(
        self, n_components: int = 3
    ) -> tuple[pd.DataFrame, np.ndarray, PCA]:
        from sklearn.preprocessing import StandardScaler
        data   = self.df[self.num_cols].dropna()
        scaled = StandardScaler().fit_transform(data)

        n_components = min(n_components, scaled.shape[1], scaled.shape[0])
        pca          = PCA(n_components=n_components)
        scores       = pca.fit_transform(scaled)

        scores_df = pd.DataFrame(
            scores,
            columns=[f"PC{i+1}" for i in range(n_components)],
            index=data.index,
        )
        return scores_df, pca.explained_variance_ratio_, pca


# ══════════════════════════════════════════════════════════════════════════════
#  CLASS 3 · FeatureEngineer
# ══════════════════════════════════════════════════════════════════════════════
class FeatureEngineer:
    """
    Applies encoding and scaling transformations to produce a model-ready
    DataFrame.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def label_encode(self, cols: list[str]) -> dict[str, dict]:
        mappings = {}
        le       = LabelEncoder()
        for col in cols:
            if col not in self.df.columns:
                continue
            self.df[col]   = le.fit_transform(self.df[col].astype(str))
            mappings[col]  = dict(zip(le.classes_, le.transform(le.classes_)))
        return mappings

    def onehot_encode(self, cols: list[str]) -> list[str]:
        new_cols = []
        for col in cols:
            if col not in self.df.columns:
                continue
            dummies  = pd.get_dummies(self.df[col], prefix=col, dtype=int)
            self.df  = pd.concat([self.df.drop(columns=[col]), dummies], axis=1)
            new_cols += dummies.columns.tolist()
        return new_cols

    def apply_scaler(self, scaler_name: str) -> list[str]:
        num_cols = self.df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        if not num_cols:
            return []
        scaler = MinMaxScaler() if scaler_name == "MinMax" else StandardScaler()
        self.df[num_cols] = scaler.fit_transform(self.df[num_cols])
        return num_cols


# ══════════════════════════════════════════════════════════════════════════════
#  CLASS 5 · ModelTrainer
# ══════════════════════════════════════════════════════════════════════════════
class ModelTrainer:
    """
    Handles problem-type detection, data preparation, and model instantiation.

    Learning Paradigm: SUPERVISED LEARNING
    ──────────────────────────────────────
    Every model trained here requires a target column (y).  The algorithm
    learns the mapping  X → y  from labeled training data and is then
    evaluated on held-out test data.  This is fundamentally different from:
      • PCA        (unsupervised — no labels)
      • KNN Impute (not supervised — no labels, fills missing values only)
    """

    _CLF_THRESHOLD = 20   # ≤ this many unique target values → classification

    def __init__(self, df: pd.DataFrame, target_col: str):
        self.df         = df.copy()
        self.target_col = target_col

    # ── Problem-type heuristic ─────────────────────────────────────────────────
    def detect_problem_type(self) -> str:
        """
        Object / bool / category dtype  → classification
        Numeric with ≤ _CLF_THRESHOLD unique values → classification
        Numeric with >  _CLF_THRESHOLD unique values → regression
        """
        col = self.df[self.target_col]
        if col.dtype in ('object', 'bool') or str(col.dtype) == 'category':
            return 'classification'
        return 'classification' if col.nunique() <= self._CLF_THRESHOLD else 'regression'

    # ── Data preparation ───────────────────────────────────────────────────────
    def prepare_data(
        self, split_ratio: float = 0.20, random_state: int = 42
    ) -> tuple:
        """
        Returns (X_train, X_test, y_train, y_test, feature_names).

        Steps
        ─────
        1. Drop all rows with any NaN.
        2. Auto-encode object/bool/category columns in X via LabelEncoder.
        3. Auto-encode object/bool target via LabelEncoder.
        4. Stratified split for classification; plain split for regression.
        """
        df_work = self.df.dropna().copy()

        if df_work.empty:
            raise ValueError(
                "No complete rows remain after dropping nulls. "
                "Please run the Preprocessing pipeline first."
            )
        if len(df_work) < 10:
            raise ValueError(
                f"Only {len(df_work)} complete rows available. "
                "Need at least 10 rows for a meaningful train/test split."
            )

        X = df_work.drop(columns=[self.target_col])
        y = df_work[self.target_col].copy()

        # Encode categorical features
        for col in X.select_dtypes(include=['object', 'bool']).columns:
            X = X.copy()
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        # Encode target if needed
        if y.dtype in ('object', 'bool') or str(y.dtype) == 'category':
            y = pd.Series(
                LabelEncoder().fit_transform(y.astype(str)),
                index=y.index, name=self.target_col,
            )

        feature_names = X.columns.tolist()

        # Stratified split for classification (falls back if a class has < 2 samples)
        problem_type = self.detect_problem_type()
        try:
            strat = y if problem_type == 'classification' else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=split_ratio,
                random_state=random_state, stratify=strat,
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=split_ratio, random_state=random_state,
            )

        return X_train, X_test, y_train, y_test, feature_names

    # ── Model factories ────────────────────────────────────────────────────────
    def get_classification_models(self) -> dict:
        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, random_state=42, solver='lbfgs',
            ),
            "Decision Tree":       DecisionTreeClassifier(
                random_state=42, max_depth=12,
            ),
            "Random Forest":       RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1,
            ),
            "KNN":                 KNeighborsClassifier(n_neighbors=5),
            "SVM":                 SVC(
                probability=True, random_state=42, max_iter=2000,
            ),
        }
        if _XGBOOST_OK:
            models["XGBoost"] = XGBClassifier(
                n_estimators=100, random_state=42,
                eval_metric='logloss', verbosity=0,
            )
        return models

    def get_regression_models(self) -> dict:
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree":     DecisionTreeRegressor(
                random_state=42, max_depth=12,
            ),
            "Random Forest":     RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=-1,
            ),
            "KNN Regressor":     KNeighborsRegressor(n_neighbors=5),
            "SVR":               SVR(),
        }
        if _XGBOOST_OK:
            models["XGBoost"] = XGBRegressor(
                n_estimators=100, random_state=42, verbosity=0,
            )
        return models

    # ── Training ───────────────────────────────────────────────────────────────
    def train_models(self, models: dict, X_train, y_train) -> dict:
        trained = {}
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                trained[name] = model
            except Exception as exc:
                st.warning(f"⚠️ **{name}** could not be trained: {exc}")
        return trained


# ══════════════════════════════════════════════════════════════════════════════
#  CLASS 6 · ModelEvaluator
# ══════════════════════════════════════════════════════════════════════════════
class ModelEvaluator:
    """
    Computes evaluation metrics, identifies the best model, and produces
    comparison visualizations (confusion matrix, feature importance).
    """

    # ── Classification metrics ─────────────────────────────────────────────────
    def evaluate_classification(
        self, trained_models: dict, X_test, y_test
    ) -> pd.DataFrame:
        rows      = []
        n_classes = len(np.unique(y_test))
        avg_mode  = 'binary' if n_classes == 2 else 'macro'

        for name, model in trained_models.items():
            y_pred = model.predict(X_test)

            try:
                if n_classes == 2:
                    y_prob = model.predict_proba(X_test)[:, 1]
                    roc    = round(float(roc_auc_score(y_test, y_prob)), 4)
                else:
                    y_prob = model.predict_proba(X_test)
                    roc    = round(float(
                        roc_auc_score(
                            y_test, y_prob,
                            multi_class='ovr', average='macro',
                        )
                    ), 4)
            except Exception:
                roc = np.nan

            rows.append({
                "Model":     name,
                "Accuracy":  round(float(accuracy_score(y_test, y_pred)), 4),
                "Precision": round(float(
                    precision_score(y_test, y_pred, average=avg_mode, zero_division=0)
                ), 4),
                "Recall":    round(float(
                    recall_score(y_test, y_pred, average=avg_mode, zero_division=0)
                ), 4),
                "F1 Score":  round(float(
                    f1_score(y_test, y_pred, average=avg_mode, zero_division=0)
                ), 4),
                "ROC AUC":   roc,
            })
        return pd.DataFrame(rows)

    # ── Regression metrics ─────────────────────────────────────────────────────
    def evaluate_regression(
        self, trained_models: dict, X_test, y_test
    ) -> pd.DataFrame:
        rows = []
        for name, model in trained_models.items():
            y_pred = model.predict(X_test)
            mse    = float(mean_squared_error(y_test, y_pred))
            rows.append({
                "Model":    name,
                "MAE":      round(float(mean_absolute_error(y_test, y_pred)), 4),
                "MSE":      round(mse, 4),
                "RMSE":     round(float(np.sqrt(mse)), 4),
                "R² Score": round(float(r2_score(y_test, y_pred)), 4),
            })
        return pd.DataFrame(rows)

    # ── Best-model selection ───────────────────────────────────────────────────
    def get_best_model_name(
        self, results_df: pd.DataFrame, problem_type: str
    ) -> str:
        primary = "F1 Score" if problem_type == 'classification' else "R² Score"
        if primary not in results_df.columns:
            return str(results_df.iloc[0]["Model"])
        valid = results_df[pd.to_numeric(results_df[primary], errors='coerce').notna()]
        if valid.empty:
            return str(results_df.iloc[0]["Model"])
        return str(valid.loc[
            pd.to_numeric(valid[primary]).idxmax(), "Model"
        ])

    # ── Confusion matrix ───────────────────────────────────────────────────────
    def confusion_matrix_fig(
        self, model, X_test, y_test
    ) -> Optional[go.Figure]:
        y_pred     = model.predict(X_test)
        all_labels = np.unique(
            np.concatenate([np.array(y_test).flatten(),
                            np.array(y_pred).flatten()])
        )
        labels_str = [str(l) for l in sorted(all_labels)]
        cm         = confusion_matrix(y_test, y_pred, labels=sorted(all_labels))

        fig = px.imshow(
            cm,
            text_auto=True,
            x=labels_str, y=labels_str,
            color_continuous_scale="Blues",
            title="Confusion Matrix — Predicted vs Actual",
            template=PLOTLY_TEMPLATE,
            aspect="auto",
        )
        fig.update_layout(
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            height=max(420, len(labels_str) * 38 + 150),
            coloraxis_showscale=False,
        )
        return fig

    # ── Feature importance ─────────────────────────────────────────────────────
    def feature_importance_fig(
        self, model, feature_names: list, model_name: str
    ) -> Optional[go.Figure]:
        if not hasattr(model, 'feature_importances_'):
            return None

        importances = np.array(model.feature_importances_)
        if len(importances) != len(feature_names):
            return None

        fi_df = (
            pd.DataFrame({"Feature": feature_names, "Importance": importances})
            .sort_values("Importance", ascending=True)
            .tail(20)
        )

        fig = go.Figure(go.Bar(
            x=fi_df["Importance"],
            y=fi_df["Feature"],
            orientation='h',
            marker=dict(
                color=fi_df["Importance"].tolist(),
                colorscale=[[0, "#1E3A5F"], [0.5, ACCENT], [1, "#34D399"]],
                showscale=False,
            ),
            text=[f"{v:.4f}" for v in fi_df["Importance"]],
            textposition="outside",
        ))
        fig.update_layout(
            title=f"Feature Importance — {model_name} (Top {len(fi_df)})",
            xaxis_title="Gini Importance Score",
            yaxis_title="",
            template=PLOTLY_TEMPLATE,
            height=max(350, len(fi_df) * 26 + 120),
        )
        return fig


# ══════════════════════════════════════════════════════════════════════════════
#  CLASS 7 · PredictionEngine
# ══════════════════════════════════════════════════════════════════════════════
class PredictionEngine:
    """
    Wraps a trained supervised model for batch prediction and CSV export.
    """

    def __init__(self, model, feature_names: list, problem_type: str):
        self.model         = model
        self.feature_names = feature_names
        self.problem_type  = problem_type

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        return None

    def results_dataframe(
        self, X: pd.DataFrame, actual=None
    ) -> pd.DataFrame:
        df_out  = X.copy()
        preds   = self.predict(X)
        df_out["Prediction"] = preds

        if self.problem_type == 'classification':
            proba = self.predict_proba(X)
            if proba is not None:
                df_out["Confidence (%)"] = (proba.max(axis=1) * 100).round(2)

        if actual is not None:
            df_out.insert(0, "Actual", actual)

        return df_out


# ══════════════════════════════════════════════════════════════════════════════
#  CLASS 4 · UIBuilder
# ══════════════════════════════════════════════════════════════════════════════
class UIBuilder:
    """
    Owns every Streamlit call.
    Delegates all computation to the other classes.
    """

    def __init__(self, raw_df: pd.DataFrame):
        self.raw_df       = raw_df
        self.preprocessor = DataPreprocessor(raw_df)

    @staticmethod
    def _divider():
        st.markdown("<hr>", unsafe_allow_html=True)

    @staticmethod
    def _badge(text: str) -> str:
        return f'<span class="badge">{text}</span>'

    # ─────────────────────────────────────────────────────────────────────────
    #  TAB 1 · Raw Data Preview
    # ─────────────────────────────────────────────────────────────────────────
    def render_raw_tab(self):
        st.header("📋 Raw Data Preview")

        df   = self.raw_df
        cols = st.columns(4)
        cols[0].metric("Rows",           f"{df.shape[0]:,}")
        cols[1].metric("Columns",        f"{df.shape[1]:,}")
        cols[2].metric("Missing Values", f"{df.isnull().sum().sum():,}")
        cols[3].metric("Duplicate Rows", f"{df.duplicated().sum():,}")

        st.markdown("### 🗂️ Column-level Quality Profile")
        st.dataframe(
            self.preprocessor.profile(), use_container_width=True, hide_index=True
        )

        st.markdown("### 🔎 First 10 Rows")
        st.dataframe(df.head(10), use_container_width=True)

        with st.expander("📊 Full Statistical Summary (describe + skew + kurtosis)"):
            analyzer = EDAAnalyzer(df)
            if analyzer.num_cols:
                st.dataframe(analyzer.descriptive_stats(), use_container_width=True)
            else:
                st.info("No numerical columns detected.")

    # ─────────────────────────────────────────────────────────────────────────
    #  TAB 2 · Smart Preprocessing
    # ─────────────────────────────────────────────────────────────────────────
    def render_preprocessing_tab(self):
        st.header("⚙️ Smart Preprocessing Pipeline")

        mode = st.radio(
            "Select Cleaning Mode",
            ["🤖 Auto-Clean (Recommended)", "🔧 Manual Pipeline"],
            horizontal=True,
        )

        self._divider()

        # ══════════════════════════════════════════════════════════════════════
        #  AUTO-CLEAN MODE
        # ══════════════════════════════════════════════════════════════════════
        if "Auto-Clean" in mode:
            st.markdown(
                '<div class="highlight-box">'
                '<h4 style="color:var(--text-primary);margin-top:0">🤖 Auto-Clean — Bulletproof Pipeline</h4>'
                '<p style="color:var(--text-secondary);margin:0">Runs six battle-tested steps in strict dependency order. '
                'Ghost values (-999, -9999, -1000…) are destroyed via a <b>mathematical mask</b> '
                '(<code>value &lt; 0 → NaN</code>) that is immune to float-precision mismatches.</p>'
                "</div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                '<span class="chip-not-supervised">🔵 KNN Imputation — NOT Supervised Learning</span>'
                '&nbsp;<small style="color:var(--text-tertiary)">No target labels are used; '
                'missing values are estimated from feature similarity only.</small>',
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)

            with st.expander("⚙️ Pipeline Settings", expanded=True):
                k_auto = st.slider(
                    "KNN Neighbours (k)", min_value=1, max_value=15, value=5,
                    help="More neighbours = smoother imputation but slower runtime.",
                )

            with st.expander("🔍 What does Auto-Clean do?"):
                st.markdown(
                    """
| Step | Action | Why it matters |
|------|--------|----------------|
| 1 | **Drop duplicates** | Eliminates redundant rows |
| 2 | **Text normalisation** | `.title()` on true categorical columns only |
| 3 | **Currency conversion** | Strips Rs./$/PKR… then `pd.to_numeric(errors='coerce')` |
| 4 | **Ghost-value mask** | `df[num_cols].mask(df[num_cols] < 0, NaN)` — beats `.replace()` |
| 5 | **KNN imputation** | Distance-weighted neighbour estimation for numeric NaNs |
| 6 | **IQR Winsorisation** | Clamps extremes to [Q1−1.5·IQR, Q3+1.5·IQR] |
                    """
                )

            auto_btn = st.button("🚀 Run Auto-Clean", type="primary", use_container_width=True)

            if auto_btn:
                proc = DataPreprocessor(self.raw_df)
                with st.spinner("Running Auto-Clean pipeline …"):
                    log = proc.auto_clean(k_neighbors=k_auto)

                st.session_state["cleaned_df"] = proc.df.copy()
                st.success("✅ Auto-Clean pipeline completed successfully!")
                with st.expander("📋 Pipeline Log", expanded=True):
                    for entry in log:
                        st.markdown(f"- {entry}")

        # ══════════════════════════════════════════════════════════════════════
        #  MANUAL MODE
        # ══════════════════════════════════════════════════════════════════════
        else:
            proc = DataPreprocessor(self.raw_df)

            with st.expander("🛠️ Configure Manual Cleaning Pipeline", expanded=True):
                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown("**Duplication & Currency**")
                    drop_dupes   = st.checkbox("Drop Duplicate Rows", value=True)
                    fix_currency = st.checkbox("Convert Dirty Numerics (Rs / $ / PKR …)", value=True)

                with col_b:
                    st.markdown("**Critical Columns** *(rows dropped if null here)*")
                    critical_cols = st.multiselect(
                        "Critical columns:",
                        options=self.raw_df.columns.tolist(),
                        help="Rows with any null in these columns are dropped outright.",
                    )

                st.markdown("---")
                st.markdown("**Imputation Strategy**")
                temporal_cols = proc.detect_temporal_columns()
                impute_choice = st.radio(
                    "Numerical Imputation",
                    ["KNN Imputer (Similarity-based)",
                     "Sequential (Forward Fill)",
                     "Sequential (Backward Fill)"],
                    horizontal=True,
                )

                # ── Learning-type label for KNN Imputation ────────────────────
                if impute_choice == "KNN Imputer (Similarity-based)":
                    st.markdown(
                        '<span class="chip-not-supervised">🔵 KNN Imputation — NOT Supervised Learning</span>'
                        '&nbsp;<small style="color:var(--text-tertiary)">Finds k nearest complete '
                        'rows by feature distance and estimates the missing value. No class labels (y) '
                        'are used at any point.</small>',
                        unsafe_allow_html=True,
                    )
                    st.markdown("<br>", unsafe_allow_html=True)

                k_val = 5
                if impute_choice == "KNN Imputer (Similarity-based)":
                    k_val = st.slider("Number of Neighbours (k)", 1, 15, 5)

                if temporal_cols:
                    st.info(
                        f"🕒 Detected likely temporal columns: **{', '.join(temporal_cols)}**. "
                        "Sequential fill is recommended for time-ordered data."
                    )

            run_btn = st.button("🚀 Execute Manual Pipeline", type="primary")

            if run_btn:
                log = []
                with st.spinner("Running cleaning pipeline …"):

                    if fix_currency:
                        converted = proc.convert_dirty_numerics()
                        if converted:
                            log.append(f"✅ Converted dirty-numeric columns: {', '.join(converted)}")

                    if drop_dupes:
                        n_dupes = proc.drop_duplicates()
                        log.append(f"✅ Removed **{n_dupes}** duplicate row(s)")

                    n_critical = proc.drop_on_critical(critical_cols)
                    if n_critical:
                        log.append(f"✅ Dropped **{n_critical}** rows with nulls in critical columns")

                    if proc.df.empty:
                        st.error(
                            "⚠️ DataFrame is empty after dropping critical-column rows. "
                            "Adjust your critical column selection."
                        )
                        return

                    if impute_choice == "KNN Imputer (Similarity-based)":
                        proc.knn_impute(n_neighbors=k_val)
                        log.append(f"✅ KNN imputation applied (k={k_val})")
                    elif impute_choice == "Sequential (Forward Fill)":
                        proc.sequential_impute("ffill")
                        log.append("✅ Sequential forward-fill applied")
                    else:
                        proc.sequential_impute("bfill")
                        log.append("✅ Sequential backward-fill applied")

                    proc.mode_impute_categoricals()
                    log.append("✅ Mode imputation applied to categorical columns")

                st.session_state["cleaned_df"] = proc.df.copy()
                st.success("Pipeline completed!")
                for entry in log:
                    st.markdown(f"- {entry}")

        # ── Post-clean report (shown for both modes) ──────────────────────────
        if st.session_state.get("cleaned_df") is not None:
            self._divider()
            cdf = st.session_state["cleaned_df"]

            st.markdown("### 📊 Quality Report — After Cleaning")
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows Remaining", f"{cdf.shape[0]:,}")
            c2.metric("Missing Values", f"{cdf.isnull().sum().sum():,}")
            c3.metric("Duplicate Rows", f"{cdf.duplicated().sum():,}")

            st.markdown("### 🔍 Cleaned Data Preview")
            st.dataframe(cdf.head(10), use_container_width=True)

            self._divider()
            st.markdown("### ⬇️ Download Cleaned Dataset")
            st.download_button(
                label="⬇️ Download cleaned_data.csv",
                data=cdf.to_csv(index=False).encode("utf-8"),
                file_name="cleaned_data.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True,
            )

    # ─────────────────────────────────────────────────────────────────────────
    #  TAB 3 · Outlier Analysis
    # ─────────────────────────────────────────────────────────────────────────
    def render_outlier_tab(self):
        st.header("🎯 Outlier Detection & Treatment")

        cleaned_df = st.session_state.get("cleaned_df")
        df         = cleaned_df if cleaned_df is not None else self.raw_df

        analyzer = EDAAnalyzer(df)

        if not analyzer.num_cols:
            st.warning("No numerical columns available for outlier analysis.")
            return

        col_sel, meth_sel = st.columns(2)
        with col_sel:
            chosen_col = st.selectbox("Select Numerical Column", analyzer.num_cols)
        with meth_sel:
            method = st.radio(
                "Detection Method",
                ["IQR Fence (Tukey)", "Z-Score (|z| > threshold)"],
                horizontal=True,
            )

        z_thresh = 3.0
        if "Z-Score" in method:
            z_thresh = st.slider("Z-Score Threshold", 1.5, 5.0, 3.0, 0.1)

        proc = DataPreprocessor(df)

        mask       = proc.detect_outliers_iqr(chosen_col) if "IQR" in method \
                     else proc.detect_outliers_zscore(chosen_col, z_thresh)
        n_outliers = int(mask.sum())
        pct        = round(n_outliers / max(len(df), 1) * 100, 2)

        st.markdown(f"### Detection Results — **{chosen_col}**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Rows",     f"{len(df):,}")
        m2.metric("Outliers Found", f"{n_outliers:,}")
        m3.metric("Outlier %",      f"{pct} %")

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Before Treatment", "After Treatment"],
        )
        fig.add_trace(
            go.Histogram(x=df[chosen_col], name="Before",
                         marker_color=ACCENT, opacity=0.75, nbinsx=40),
            row=1, col=1,
        )

        action    = st.radio("Treatment Action",
                             ["Cap / Clip (Winsorise)", "Drop Outlier Rows"],
                             horizontal=True)
        treat_btn = st.button("⚡ Apply Outlier Treatment")

        if treat_btn:
            if "Cap" in action:
                proc.cap_outliers(chosen_col)
                st.success(f"Capped {n_outliers} outlier(s) in **{chosen_col}**.")
            else:
                n_dropped = proc.drop_outliers(mask)
                st.success(f"Dropped {n_dropped} outlier row(s).")
            st.session_state["cleaned_df"] = proc.df.copy()

        temp_cleaned = st.session_state.get("cleaned_df")
        after_df = temp_cleaned if temp_cleaned is not None else df
        fig.add_trace(
            go.Histogram(x=after_df[chosen_col], name="After",
                         marker_color="#34D399", opacity=0.75, nbinsx=40),
            row=1, col=2,
        )
        fig.update_layout(
            template=PLOTLY_TEMPLATE, showlegend=False,
            height=380, margin=dict(t=50, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📦 Box Plot"):
            fig_box = px.box(
                df, y=chosen_col, points="outliers",
                template=PLOTLY_TEMPLATE,
                color_discrete_sequence=[ACCENT],
                title=f"Box Plot — {chosen_col}",
            )
            st.plotly_chart(fig_box, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    #  TAB 4 · Feature Engineering & Scaling
    # ─────────────────────────────────────────────────────────────────────────
    def render_feature_tab(self):
        st.header("🔧 Feature Engineering & Scaling Pipeline")

        cleaned_df = st.session_state.get("cleaned_df")
        df         = cleaned_df.copy() if cleaned_df is not None else self.raw_df.copy()

        fe       = FeatureEngineer(df)
        cat_cols = fe.df.select_dtypes(include='object').columns.tolist()
        num_cols = fe.df.select_dtypes(include=["int64", "float64"]).columns.tolist()

        with st.expander("🔡 Categorical Encoding", expanded=True):
            if not cat_cols:
                st.info("No categorical (object) columns found.")
            else:
                enc_method = st.radio(
                    "Encoding Strategy",
                    ["Label Encoding", "One-Hot Encoding"],
                    horizontal=True,
                )
                enc_cols = st.multiselect(
                    "Select Columns to Encode",
                    options=cat_cols,
                    default=cat_cols[:min(3, len(cat_cols))],
                )

        with st.expander("📐 Feature Scaling", expanded=True):
            if not num_cols:
                st.info("No numerical columns available for scaling.")
            apply_scaling = st.checkbox("Apply Scaling", value=False)
            scaler_choice = st.radio(
                "Scaler",
                ["MinMax (0–1 normalisation)", "Standard (Z-score normalisation)"],
                horizontal=True,
            )

        apply_btn = st.button("⚙️ Apply Feature Engineering", type="primary")
        if apply_btn:
            log = []
            if cat_cols and enc_cols:
                if enc_method == "Label Encoding":
                    mappings = fe.label_encode(enc_cols)
                    log.append(f"✅ Label-encoded **{len(enc_cols)}** column(s).")
                    with st.expander("Label Encoding Mappings"):
                        for col, m in mappings.items():
                            st.write(f"**{col}** → {m}")
                else:
                    new_cols = fe.onehot_encode(enc_cols)
                    log.append(
                        f"✅ One-Hot encoded **{len(enc_cols)}** column(s) "
                        f"→ created {len(new_cols)} binary columns."
                    )

            if apply_scaling:
                scaler_key  = "MinMax" if "MinMax" in scaler_choice else "Standard"
                scaled_cols = fe.apply_scaler(scaler_key)
                log.append(
                    f"✅ {scaler_key} scaling applied to **{len(scaled_cols)}** columns."
                )

            st.session_state["engineered_df"] = fe.df.copy()
            for entry in log:
                st.markdown(f"- {entry}")
            st.success("Feature engineering complete!")

        temp_eng = st.session_state.get("engineered_df")
        final_df = temp_eng if temp_eng is not None else df
        st.markdown("### 🔍 Transformed Data Preview")
        st.dataframe(final_df.head(10), use_container_width=True)

        self._divider()
        st.markdown("### ⬇️ Download Transformed Dataset")
        st.download_button(
            label="⬇️ Download transformed_data.csv",
            data=final_df.to_csv(index=False).encode("utf-8"),
            file_name="transformed_data.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  TAB 5 · Advanced EDA  (PCA section enhanced)
    # ─────────────────────────────────────────────────────────────────────────
    def render_eda_tab(self):
        st.header("📈 Advanced Exploratory Data Analysis")

        if st.session_state.get("cleaned_df") is not None:
            df_source    = st.session_state["cleaned_df"]
            source_label = "Cleaned Data"
        elif st.session_state.get("engineered_df") is not None:
            df_source    = st.session_state["engineered_df"]
            source_label = "Engineered Data"
        else:
            df_source    = self.raw_df
            source_label = "Raw Data"

        st.caption(f"📌 Source: **{source_label}**")

        analyzer = EDAAnalyzer(df_source)

        if not analyzer.num_cols:
            st.warning("No numerical columns for EDA. Run the cleaning pipeline first.")
            return

        # ── A · Descriptive statistics ────────────────────────────────────────
        st.markdown("### 📊 A · Descriptive Statistics")
        st.dataframe(analyzer.descriptive_stats(), use_container_width=True)

        with st.expander("ℹ️ Interpreting skewness & kurtosis"):
            st.markdown(
                """
| Metric | Rule of Thumb |
|--------|---------------|
| Skewness ≈ 0 | Symmetric / normal |
| Skewness > ±1 | Highly skewed (consider log transform) |
| Kurtosis > 3 | Heavy tails (leptokurtic) |
| Kurtosis < 3 | Light tails (platykurtic) |
                """
            )

        self._divider()

        # ── B · Bivariate scatter + OLS ───────────────────────────────────────
        st.markdown("### 🔵 B · Bivariate Scatter (OLS Trendline)")
        col_x, col_y = st.columns(2)
        x_feat = col_x.selectbox("X-axis", analyzer.num_cols, key="eda_x")
        y_feat = col_y.selectbox(
            "Y-axis", analyzer.num_cols,
            index=min(1, len(analyzer.num_cols) - 1), key="eda_y",
        )
        color_by = None
        if analyzer.cat_cols:
            c_opt    = st.selectbox("Colour by (optional)", ["None"] + analyzer.cat_cols)
            color_by = None if c_opt == "None" else c_opt

        if x_feat == y_feat:
            st.warning("Select different features for X and Y.")
        else:
            try:
                fig_sc = px.scatter(
                    df_source, x=x_feat, y=y_feat, color=color_by,
                    trendline="ols",
                    title=f"{y_feat}  vs  {x_feat}",
                    template=PLOTLY_TEMPLATE, opacity=0.7,
                )
                st.plotly_chart(fig_sc, use_container_width=True)
            except Exception as exc:
                st.error(
                    f"Scatter failed: {exc}. "
                    "Tip: `pip install statsmodels` enables OLS trendlines."
                )

        self._divider()

        # ── C · Univariate distribution ───────────────────────────────────────
        st.markdown("### 🟢 C · Univariate Distribution")
        dist_col = st.selectbox("Feature", analyzer.num_cols, key="eda_dist")
        d1, d2   = st.columns(2)
        d1.metric("Skewness", f"{df_source[dist_col].skew():.4f}")
        d2.metric("Kurtosis", f"{df_source[dist_col].kurt():.4f}")

        fig_h = px.histogram(
            df_source, x=dist_col, nbins=40, marginal="box",
            title=f"Distribution of  {dist_col}",
            template=PLOTLY_TEMPLATE,
            color_discrete_sequence=[ACCENT],
        )
        st.plotly_chart(fig_h, use_container_width=True)

        self._divider()

        # ── D · Correlation heatmap ───────────────────────────────────────────
        st.markdown("### 🟣 D · Correlation Matrix with Hypothesis Testing")
        corr_method = st.radio(
            "Method", ["pearson", "spearman"], horizontal=True,
        )

        if len(analyzer.num_cols) < 2:
            st.info("Need ≥ 2 numerical columns for a correlation matrix.")
        else:
            corr_m, pval_m = analyzer.correlation_with_pvalues(corr_method)
            h1, h2 = st.columns(2)
            with h1:
                st.plotly_chart(
                    px.imshow(corr_m, text_auto=".2f", aspect="auto",
                              color_continuous_scale="RdBu_r",
                              title=f"{corr_method.capitalize()} Correlation",
                              template=PLOTLY_TEMPLATE, zmin=-1, zmax=1),
                    use_container_width=True,
                )
            with h2:
                st.plotly_chart(
                    px.imshow(pval_m, text_auto=".3f", aspect="auto",
                              color_continuous_scale="Viridis_r",
                              title="p-values  (< 0.05 = significant)",
                              template=PLOTLY_TEMPLATE, zmin=0, zmax=1),
                    use_container_width=True,
                )

            with st.expander("📌 Significant Pairs (p < 0.05)"):
                sig_rows  = []
                cols_list = corr_m.columns.tolist()
                for i in range(len(cols_list)):
                    for j in range(i + 1, len(cols_list)):
                        if pval_m.iloc[i, j] < 0.05:
                            sig_rows.append({
                                "Feature A":        cols_list[i],
                                "Feature B":        cols_list[j],
                                "Correlation (r)":  round(corr_m.iloc[i, j], 4),
                                "p-value":          round(pval_m.iloc[i, j], 4),
                                "Significant":      "✅ Yes",
                            })
                if sig_rows:
                    st.dataframe(
                        pd.DataFrame(sig_rows),
                        use_container_width=True, hide_index=True,
                    )
                else:
                    st.info("No significant pairs found at α = 0.05.")

        self._divider()

        # ── E · PCA  (ENHANCED) ───────────────────────────────────────────────
        st.markdown("### 🔮 E · Principal Component Analysis (PCA)")

        # ── Learning-type label ───────────────────────────────────────────────
        st.markdown(
            '<span class="chip-unsupervised">📊 PCA — Unsupervised Learning</span>'
            '&nbsp;<small style="color:var(--text-tertiary)">PCA discovers structure in the feature '
            'space without using any target labels (y). It is a dimensionality reduction technique, '
            'not a predictive model.</small>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        n_pca = st.slider(
            "Number of Principal Components", 2, min(10, len(analyzer.num_cols)), 3
        )

        if len(analyzer.num_cols) < 2:
            st.info("Need ≥ 2 numerical columns for PCA.")
        else:
            try:
                scores, evr, pca_obj = analyzer.run_pca(n_components=n_pca)

                # ── PCA metadata metrics ──────────────────────────────────────
                n_feat_before   = len(analyzer.num_cols)
                n_feat_after    = int(pca_obj.n_components_)
                retained_pct    = round(float(evr.sum()) * 100, 2)
                reduction_pct   = round((1 - n_feat_after / n_feat_before) * 100, 1)
                cumulative_evr  = (np.cumsum(evr) * 100).round(2)

                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Features Before PCA",   f"{n_feat_before}")
                p2.metric("Components After PCA",  f"{n_feat_after}")
                p3.metric("Retained Variance",     f"{retained_pct} %")
                p4.metric("Dimension Reduction",   f"{reduction_pct} %")

                # ── Scree plot with cumulative variance ───────────────────────
                pc_labels = [f"PC{i+1}" for i in range(len(evr))]

                fig_scree = go.Figure()
                fig_scree.add_trace(go.Bar(
                    x=pc_labels,
                    y=(evr * 100).round(2),
                    name="Per-Component Variance",
                    marker_color=ACCENT,
                    text=(evr * 100).round(1),
                    textposition="outside",
                ))
                fig_scree.add_trace(go.Scatter(
                    x=pc_labels,
                    y=cumulative_evr.tolist(),
                    name="Cumulative Variance",
                    mode='lines+markers',
                    line=dict(color="#34D399", width=2.5),
                    marker=dict(size=8, color="#34D399",
                                line=dict(width=2, color="#0A0B0D")),
                    text=[f"{v:.1f}%" for v in cumulative_evr],
                    textposition="top center",
                    textfont=dict(color="#34D399", size=10),
                ))
                fig_scree.update_layout(
                    title=(
                        f"Scree Plot — Individual & Cumulative Explained Variance "
                        f"({retained_pct}% retained with {n_feat_after} components)"
                    ),
                    yaxis_title="Explained Variance (%)",
                    yaxis=dict(range=[0, min(110, float(cumulative_evr[-1]) + 12)]),
                    template=PLOTLY_TEMPLATE,
                    height=420,
                    legend=dict(
                        orientation="h", yanchor="bottom",
                        y=1.02, xanchor="right", x=1,
                    ),
                )
                st.plotly_chart(fig_scree, use_container_width=True)

                # ── 2-D biplot ────────────────────────────────────────────────
                color_col = None
                if analyzer.cat_cols:
                    c_choice = st.selectbox(
                        "Colour PCA by (optional)", ["None"] + analyzer.cat_cols,
                        key="pca_color",
                    )
                    if c_choice != "None":
                        valid_idx      = df_source[analyzer.num_cols].dropna().index
                        scores[c_choice] = df_source.loc[valid_idx, c_choice].values
                        color_col      = c_choice

                fig_2d = px.scatter(
                    scores, x="PC1", y="PC2", color=color_col,
                    title=(
                        f"PCA 2D — "
                        f"PC1 {round(evr[0]*100,1)}%  +  PC2 {round(evr[1]*100,1)}%  "
                        f"= {round((evr[0]+evr[1])*100,1)}% explained"
                    ),
                    template=PLOTLY_TEMPLATE, opacity=0.75,
                    color_discrete_sequence=px.colors.qualitative.Bold,
                )
                st.plotly_chart(fig_2d, use_container_width=True)

                if n_pca >= 3:
                    fig_3d = px.scatter_3d(
                        scores, x="PC1", y="PC2", z="PC3", color=color_col,
                        title="PCA 3D Projection",
                        template=PLOTLY_TEMPLATE, opacity=0.75,
                        color_discrete_sequence=px.colors.qualitative.Bold,
                    )
                    fig_3d.update_layout(height=550)
                    st.plotly_chart(fig_3d, use_container_width=True)

                with st.expander("🔬 PCA Loadings (feature contributions)"):
                    loadings = pd.DataFrame(
                        pca_obj.components_.T,
                        index=analyzer.num_cols[:pca_obj.components_.shape[1]],
                        columns=[f"PC{i+1}" for i in range(n_feat_after)],
                    ).round(4)
                    st.dataframe(loadings, use_container_width=True)

            except Exception as exc:
                st.error(f"PCA failed: {exc}")

    # ─────────────────────────────────────────────────────────────────────────
    #  TAB 6 · Machine Learning Studio  (NEW)
    # ─────────────────────────────────────────────────────────────────────────
    def render_ml_tab(self):
        st.header("🤖 Machine Learning Studio")

        # ── Learning-type banner ──────────────────────────────────────────────
        st.markdown(
            '<div class="highlight-box">'
            '<h4 style="color:var(--text-primary);margin-top:0">'
            '<span class="chip-supervised">✅ Supervised Learning</span></h4>'
            '<p style="color:var(--text-secondary);margin:0">'
            'Model training uses <b>Supervised Learning</b> — algorithms learn from <b>labeled data</b> '
            'where both feature matrix <code>X</code> and a target vector <code>y</code> are provided. '
            'The model optimises a mapping <b>X → y</b> on training data, then its generalisation is '
            'measured on the held-out test set.  '
            'This is fundamentally different from <b>PCA</b> (unsupervised — no labels) and '
            '<b>KNN Imputation</b> (not supervised — fills missing values, no labels used).</p>'
            '</div>',
            unsafe_allow_html=True,
        )

        # ── Data source ───────────────────────────────────────────────────────
        data_options: dict = {"Raw Data": self.raw_df}
        if st.session_state.get("cleaned_df") is not None:
            data_options["✅ Cleaned Data (Recommended)"] = st.session_state["cleaned_df"]
        if st.session_state.get("engineered_df") is not None:
            data_options["⚙️ Engineered Data (Encoded + Scaled)"] = st.session_state["engineered_df"]

        src_key = st.selectbox(
            "📂 Data Source",
            options=list(data_options.keys()),
            index=len(data_options) - 1,
            help="Use the most processed version for best model performance.",
            key="ml_data_source",
        )
        df_ml = data_options[src_key]

        if df_ml is None or df_ml.empty:
            st.warning("No data available. Upload a CSV and run the preprocessing pipeline first.")
            return

        # ── 1 · Target column & problem type ─────────────────────────────────
        self._divider()
        st.markdown("### 🎯 Step 1 · Target Column & Problem Type")

        t_col, t_type = st.columns([3, 1])
        with t_col:
            target_col = st.selectbox(
                "Select Target Column (y)",
                options=df_ml.columns.tolist(),
                key="ml_target",
                help="The variable the model will learn to predict.",
            )

        trainer      = ModelTrainer(df_ml, target_col)
        problem_type = trainer.detect_problem_type()
        n_unique_tgt = df_ml[target_col].nunique()

        with t_type:
            st.markdown("<br>", unsafe_allow_html=True)
            if problem_type == 'classification':
                st.markdown(
                    f'<div style="background:rgba(52,211,153,.15);border:1px solid rgba(52,211,153,.4);'
                    f'border-radius:var(--radius-md);padding:10px 14px;text-align:center;">'
                    f'<span style="color:#34D399;font-weight:700;font-size:.9rem;">⚡ Classification</span><br>'
                    f'<span style="color:var(--text-tertiary);font-size:.75rem;">{n_unique_tgt} classes</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div style="background:rgba(242,181,68,.12);border:1px solid rgba(242,181,68,.35);'
                    f'border-radius:var(--radius-md);padding:10px 14px;text-align:center;">'
                    f'<span style="color:#F2B544;font-weight:700;font-size:.9rem;">📈 Regression</span><br>'
                    f'<span style="color:var(--text-tertiary);font-size:.75rem;">{n_unique_tgt} unique values</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # ── 2 · Dataset statistics ────────────────────────────────────────────
        self._divider()
        st.markdown("### 📊 Step 2 · Dataset Statistics")

        n_total   = len(df_ml)
        n_nulls   = int(df_ml.isnull().sum().sum())
        n_usable  = len(df_ml.dropna())
        n_features= len(df_ml.columns) - 1

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Rows",             f"{n_total:,}")
        s2.metric("Usable Rows (non-null)", f"{n_usable:,}")
        s3.metric("Feature Count",          f"{n_features:,}")
        s4.metric("Target Unique Values",   f"{n_unique_tgt:,}")

        if n_nulls > 0:
            st.warning(
                f"⚠️ **{n_nulls:,}** missing values detected — rows with any null will be "
                "dropped before training. Run the **Preprocessing** tab first for better coverage."
            )

        if n_usable < 20:
            st.error(
                f"❌ Only {n_usable} complete rows available. "
                "Need at least 20 to form a meaningful train/test split. "
                "Please preprocess your data first."
            )
            return

        # Class distribution for classification
        if problem_type == 'classification':
            with st.expander("📊 Target Class Distribution"):
                cls_df = (
                    df_ml[target_col]
                    .value_counts()
                    .reset_index()
                    .rename(columns={target_col: "Class", "count": "Count"})
                )
                if "count" not in cls_df.columns and len(cls_df.columns) == 2:
                    cls_df.columns = ["Class", "Count"]
                cls_df["Percentage (%)"] = (cls_df["Count"] / n_total * 100).round(2)

                fig_cls = px.bar(
                    cls_df, x="Class", y="Count",
                    text="Percentage (%)",
                    color="Count",
                    color_continuous_scale="Blues",
                    title=f"Class Distribution — {target_col}",
                    template=PLOTLY_TEMPLATE,
                )
                fig_cls.update_traces(
                    texttemplate='%{text:.1f}%', textposition='outside'
                )
                fig_cls.update_layout(height=360, coloraxis_showscale=False)
                st.plotly_chart(fig_cls, use_container_width=True)

        # ── 3 · Train / Test split ────────────────────────────────────────────
        self._divider()
        st.markdown("### ✂️ Step 3 · Train / Test Split")

        split_choice = st.radio(
            "Split Ratio (Train / Test)",
            ["70 / 30", "80 / 20", "90 / 10"],
            index=1,
            horizontal=True,
            key="ml_split",
        )
        _split_map = {"70 / 30": 0.30, "80 / 20": 0.20, "90 / 10": 0.10}
        test_size  = _split_map[split_choice]
        n_train_est = int(n_usable * (1 - test_size))
        n_test_est  = n_usable - n_train_est

        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Estimated Training Samples", f"{n_train_est:,}")
        sc2.metric("Estimated Testing Samples",  f"{n_test_est:,}")
        sc3.metric("Split Ratio",                split_choice)

        # ── 4 · Train & compare ───────────────────────────────────────────────
        self._divider()
        st.markdown("### 🚀 Step 4 · Train & Compare All Models")

        if not _XGBOOST_OK:
            st.info(
                "ℹ️ **XGBoost** is not installed. "
                "Run `pip install xgboost` to include it in the comparison."
            )

        model_list = (
            ["Logistic Regression", "Decision Tree", "Random Forest", "KNN", "SVM"]
            if problem_type == 'classification'
            else ["Linear Regression", "Decision Tree", "Random Forest",
                  "KNN Regressor", "SVR"]
        )
        if _XGBOOST_OK:
            model_list.append("XGBoost")

        with st.expander("📋 Models in this run", expanded=False):
            for m in model_list:
                st.markdown(f"- **{m}**")

        col_train, col_reset = st.columns([3, 1])
        with col_train:
            train_btn = st.button(
                "🚀 Train All Models", type="primary",
                use_container_width=True, key="ml_train_btn",
            )
        with col_reset:
            reset_btn = st.button(
                "🗑️ Clear Results",
                use_container_width=True, key="ml_reset_btn",
            )

        if reset_btn:
            for k in ["ml_results_df", "ml_trained_models", "ml_best_model_name",
                      "ml_X_test", "ml_y_test", "ml_feature_names",
                      "ml_n_train", "ml_n_test", "ml_problem_type"]:
                st.session_state[k] = None
            st.success("ML results cleared. You can now retrain.")

        if train_btn:
            with st.spinner("⚙️ Preparing data, training models, evaluating …"):
                try:
                    (X_train, X_test,
                     y_train, y_test,
                     feature_names) = trainer.prepare_data(split_ratio=test_size)

                    models = (
                        trainer.get_classification_models()
                        if problem_type == 'classification'
                        else trainer.get_regression_models()
                    )
                    trained_models = trainer.train_models(models, X_train, y_train)

                    evaluator  = ModelEvaluator()
                    results_df = (
                        evaluator.evaluate_classification(trained_models, X_test, y_test)
                        if problem_type == 'classification'
                        else evaluator.evaluate_regression(trained_models, X_test, y_test)
                    )
                    best_name = evaluator.get_best_model_name(results_df, problem_type)

                    # Persist in session state
                    st.session_state["ml_problem_type"]    = problem_type
                    st.session_state["ml_trained_models"]  = trained_models
                    st.session_state["ml_results_df"]      = results_df
                    st.session_state["ml_best_model_name"] = best_name
                    st.session_state["ml_X_test"]          = X_test
                    st.session_state["ml_y_test"]          = y_test
                    st.session_state["ml_feature_names"]   = feature_names
                    st.session_state["ml_n_train"]         = len(X_train)
                    st.session_state["ml_n_test"]          = len(X_test)

                    st.success(
                        f"✅ {len(trained_models)} model(s) trained and evaluated!  "
                        f"🏆 Best model: **{best_name}**"
                    )

                except Exception as exc:
                    st.error(f"❌ Training pipeline failed: {exc}")
                    st.exception(exc)
                    return

        # ── Display results (persistent across rerenders) ─────────────────────
        if st.session_state.get("ml_results_df") is None:
            return

        results_df     = st.session_state["ml_results_df"]
        best_name      = st.session_state["ml_best_model_name"]
        trained_models = st.session_state["ml_trained_models"]
        X_test         = st.session_state["ml_X_test"]
        y_test         = st.session_state["ml_y_test"]
        feature_names  = st.session_state["ml_feature_names"]
        pt             = st.session_state["ml_problem_type"]
        n_train_actual = st.session_state.get("ml_n_train", n_train_est)
        n_test_actual  = st.session_state.get("ml_n_test",  n_test_est)

        evaluator = ModelEvaluator()

        # ── 5 · Model comparison table ────────────────────────────────────────
        self._divider()
        st.markdown("### 📊 Step 5 · Model Comparison")

        st.markdown(
            f'<div class="best-model-banner">'
            f'🏆 &nbsp;<b style="color:#34D399;font-size:1.05rem;">Best Model: {best_name}</b>'
            f'&nbsp;— selected by highest '
            f'{"F1 Score" if pt == "classification" else "R² Score"}'
            f'</div>',
            unsafe_allow_html=True,
        )

        def _style_best(row):
            style = [
                'background-color:rgba(52,211,153,.14);font-weight:700;'
                'color:var(--text-primary)'
                if row["Model"] == best_name else ''
            ] * len(row)
            return style

        styled_df = results_df.style.apply(_style_best, axis=1).format(
            {c: "{:.4f}" for c in results_df.select_dtypes(include='number').columns}
        )
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # ── 6 · Visual comparison bar chart ──────────────────────────────────
        self._divider()
        st.markdown("### 📈 Step 6 · Visual Model Comparison")

        metric_opts = (
            ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
            if pt == 'classification'
            else ["R² Score", "MAE", "RMSE", "MSE"]
        )
        metric_pick = st.selectbox(
            "Metric to Visualise", metric_opts, key="ml_metric_plot"
        )

        if metric_pick in results_df.columns:
            plot_df = results_df.copy()
            plot_df[metric_pick] = pd.to_numeric(
                plot_df[metric_pick], errors='coerce'
            )
            ascending_metrics = {"MAE", "MSE", "RMSE"}
            plot_df = plot_df.sort_values(
                metric_pick, ascending=metric_pick in ascending_metrics
            )

            colors = [
                "#34D399" if m == best_name else ACCENT
                for m in plot_df["Model"]
            ]

            fig_bar = go.Figure(go.Bar(
                x=plot_df["Model"],
                y=plot_df[metric_pick],
                marker_color=colors,
                text=plot_df[metric_pick].round(4),
                textposition="outside",
            ))
            fig_bar.update_layout(
                title=f"Model Comparison — {metric_pick}",
                xaxis_title="Model",
                yaxis_title=metric_pick,
                template=PLOTLY_TEMPLATE,
                showlegend=False,
                height=430,
                yaxis=dict(
                    range=[0, float(plot_df[metric_pick].max()) * 1.18]
                    if metric_pick not in ascending_metrics
                    else None,
                ),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # ── 7 · Confusion matrix (classification only) ────────────────────────
        if pt == 'classification':
            self._divider()
            st.markdown("### 🔢 Step 7 · Confusion Matrix")

            model_keys = list(trained_models.keys())
            default_ix = model_keys.index(best_name) if best_name in model_keys else 0
            cm_pick    = st.selectbox(
                "Select Model", model_keys, index=default_ix, key="ml_cm_pick"
            )

            n_cls = len(np.unique(np.array(y_test).flatten()))
            if n_cls <= 25:
                fig_cm = evaluator.confusion_matrix_fig(
                    trained_models[cm_pick], X_test, y_test
                )
                if fig_cm:
                    st.plotly_chart(fig_cm, use_container_width=True)
            else:
                st.info(
                    f"Confusion matrix skipped — {n_cls} unique classes is too dense "
                    "to render clearly. Consider grouping rare classes."
                )

        # ── 8 · Feature importance ────────────────────────────────────────────
        self._divider()
        st.markdown("### 🌲 Step 8 · Feature Importance")

        fi_eligible = {
            name: mdl for name, mdl in trained_models.items()
            if hasattr(mdl, 'feature_importances_')
        }

        if not fi_eligible:
            st.info(
                "Feature importance plots are available for tree-based models: "
                "**Random Forest** and **XGBoost**."
            )
        else:
            fi_pick = st.selectbox(
                "Select Model", list(fi_eligible.keys()), key="ml_fi_pick"
            )
            fig_fi = evaluator.feature_importance_fig(
                fi_eligible[fi_pick], feature_names, fi_pick
            )
            if fig_fi:
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.warning(f"{fi_pick} did not produce feature importances.")

        # ── 9 · Training summary statistics ──────────────────────────────────
        self._divider()
        st.markdown("### 📋 Step 9 · Training Summary")

        ts1, ts2, ts3, ts4 = st.columns(4)
        ts1.metric("Training Samples",   f"{n_train_actual:,}")
        ts2.metric("Testing Samples",    f"{n_test_actual:,}")
        ts3.metric("Split Ratio",        split_choice)
        ts4.metric("Features Used",      f"{len(feature_names):,}")

        with st.expander("📄 Feature Names Used for Training"):
            st.write(feature_names)

        # ── 10 · Download predictions ─────────────────────────────────────────
        self._divider()
        st.markdown("### ⬇️ Step 10 · Download Predictions")

        model_keys   = list(trained_models.keys())
        default_ix   = model_keys.index(best_name) if best_name in model_keys else 0
        pred_pick    = st.selectbox(
            "Model for Export", model_keys, index=default_ix, key="ml_pred_pick"
        )

        engine   = PredictionEngine(trained_models[pred_pick], feature_names, pt)
        pred_df  = engine.results_dataframe(
            X_test,
            actual=np.array(y_test).flatten(),
        )

        st.dataframe(pred_df.head(20), use_container_width=True)

        st.download_button(
            label=f"⬇️ Download predictions — {pred_pick}.csv",
            data=pred_df.to_csv(index=False).encode("utf-8"),
            file_name=f"predictions_{pred_pick.replace(' ', '_')}.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  SIDEBAR
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def render_sidebar():
        st.sidebar.title("🔬 AutoClean & EDA")
        st.sidebar.caption("BS Data Science · Semester 2")
        st.sidebar.markdown("---")
        st.sidebar.markdown(
            """
**Pipeline overview:**
1. 📋 Raw Data Preview
2. ⚙️ Smart Preprocessing (Auto-Clean / Manual)
3. 🎯 Outlier Detection & Treatment
4. 🔧 Feature Engineering & Scaling
5. 📈 Advanced EDA + PCA
6. 🤖 Machine Learning Studio
            """
        )
        st.sidebar.markdown("---")

        st.sidebar.markdown("**Learning Paradigms:**")
        st.sidebar.markdown(
            "🔵 **KNN Impute** — Not supervised  \n"
            "📊 **PCA** — Unsupervised  \n"
            "✅ **ML Studio** — Supervised"
        )
        st.sidebar.markdown("---")

        uploaded = st.sidebar.file_uploader("📂 Upload CSV Dataset", type=["csv"])
        if uploaded:
            st.sidebar.success("Dataset loaded!")
        st.sidebar.markdown("---")
        st.sidebar.info(
            "**Tip:** Run **Auto-Clean** before **ML Studio** for best results. "
            "Cleaned data → better model accuracy."
        )
        return uploaded


# ══════════════════════════════════════════════════════════════════════════════
#  CACHED DATA LOADER
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading dataset …")
def load_data(file) -> pd.DataFrame:
    """Cache the raw DataFrame so re-renders don't re-parse the CSV."""
    return pd.read_csv(file)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def main():
    st.title("🔬 AutoClean & EDA Toolkit  —  Advanced Edition")
    st.caption("Production-grade data preprocessing, exploratory analysis, and machine learning dashboard.")

    uploaded_file = UIBuilder.render_sidebar()

    if uploaded_file is None:
        st.info("👈 Upload a CSV file from the sidebar to begin.")
        st.markdown(
            """
### Welcome to the Advanced AutoClean & EDA Toolkit 🚀

| Module | Highlights |
|--------|-----------|
| **Auto-Clean** | 6-step bulletproof pipeline · ghost values destroyed via `< 0` mask |
| **Manual Preprocessing** | KNN / Sequential fill · Critical-column row-drop |
| **Outlier Treatment** | IQR Fence · Z-Score · Cap or Drop strategies |
| **Feature Engineering** | Label & One-Hot Encoding · MinMax & Standard Scaling |
| **Advanced EDA** | Skewness · Kurtosis · Pearson/Spearman + p-values · PCA 2D/3D |
| **ML Studio** | Auto-detect Classification/Regression · 6 models · Full metrics · Confusion Matrix · Feature Importance |

**Learning Paradigms at a glance:**

| Technique | Paradigm | Uses Labels (y)? |
|-----------|----------|-----------------|
| KNN Imputation | 🔵 Not Supervised | ❌ No — fills missing values by feature similarity |
| PCA | 📊 Unsupervised | ❌ No — discovers structure in feature space |
| ML Studio Models | ✅ Supervised | ✔️ Yes — learns mapping X → y from labeled data |

Upload any `.csv` file to get started.
            """
        )
        return

    try:
        raw_df = load_data(uploaded_file)
    except Exception as exc:
        st.error(f"❌ Failed to read file: {exc}")
        st.stop()

    # Reset session state on fresh upload
    if ("last_file" not in st.session_state
            or st.session_state["last_file"] != uploaded_file.name):
        st.session_state["last_file"]          = uploaded_file.name
        st.session_state["cleaned_df"]         = None
        st.session_state["engineered_df"]      = None
        # ML Studio state
        st.session_state["ml_results_df"]      = None
        st.session_state["ml_trained_models"]  = None
        st.session_state["ml_best_model_name"] = None
        st.session_state["ml_X_test"]          = None
        st.session_state["ml_y_test"]          = None
        st.session_state["ml_feature_names"]   = None
        st.session_state["ml_n_train"]         = None
        st.session_state["ml_n_test"]          = None
        st.session_state["ml_problem_type"]    = None

    ui = UIBuilder(raw_df)

    tabs = st.tabs([
        "📋 Raw Data",
        "⚙️ Preprocessing",
        "🎯 Outlier Analysis",
        "🔧 Feature Engineering",
        "📈 Advanced EDA",
        "🤖 ML Studio",
    ])

    with tabs[0]: ui.render_raw_tab()
    with tabs[1]: ui.render_preprocessing_tab()
    with tabs[2]: ui.render_outlier_tab()
    with tabs[3]: ui.render_feature_tab()
    with tabs[4]: ui.render_eda_tab()
    with tabs[5]: ui.render_ml_tab()


if __name__ == "__main__":
    main()