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
  │  UIBuilder         – all Streamlit rendering logic      │
  └─────────────────────────────────────────────────────────┘

  KEY FIX (auto_clean):
  Ghost values like -999 / -9999 are destroyed via a mathematical mask
  (df[num_cols] < 0) → np.nan, which is immune to float-precision issues
  that cause df.replace(-999) to silently miss converted float cells.
"""

# ── Standard library ──────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

# ── Third-party ───────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import (
    MinMaxScaler, StandardScaler, LabelEncoder,
)
from scipy.stats import zscore, pearsonr, spearmanr

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
/* ── Base ──────────────────────────────────────────────── */
.stApp                { background:#080C14; color:#E2E8F0; }
section[data-testid="stSidebar"] { background:#0D1117; }

/* ── Metric cards ──────────────────────────────────────── */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg,#131B2E 0%,#1A2540 100%);
    border:1px solid #2A3A5C; border-radius:12px;
    padding:18px 22px; box-shadow:0 4px 20px rgba(0,0,0,.4);
}
div[data-testid="stMetricLabel"] p { color:#7DD3FC; font-size:.85rem; }
div[data-testid="stMetricValue"]   { color:#F0F9FF; font-weight:700; }

/* ── Headings ──────────────────────────────────────────── */
h1 { color:#38BDF8; letter-spacing:-0.5px; }
h2 { color:#7DD3FC; }
h3 { color:#BAE6FD; }

/* ── Tabs ──────────────────────────────────────────────── */
button[data-baseweb="tab"]                          { font-size:15px; font-weight:600; color:#94A3B8; }
button[data-baseweb="tab"][aria-selected="true"]    { color:#38BDF8; border-bottom:2px solid #38BDF8; }

/* ── Expanders ─────────────────────────────────────────── */
details summary { background:#131B2E; border-radius:8px; padding:10px 14px; color:#7DD3FC; }

/* ── Alerts ────────────────────────────────────────────── */
div[data-testid="stAlert"] { border-radius:10px; }

/* ── Dataframe ─────────────────────────────────────────── */
.stDataFrame { border-radius:10px; overflow:hidden; }

/* ── Divider ───────────────────────────────────────────── */
hr { border-color:#1E293B; }

/* ── Badge ─────────────────────────────────────────────── */
.badge {
    display:inline-block; background:#14532D; color:#86EFAC;
    border:1px solid #22C55E; border-radius:999px;
    padding:3px 12px; font-size:.78rem; font-weight:600; margin-left:8px;
}
.highlight-box {
    background:linear-gradient(135deg,#1A1040 0%,#12172B 100%);
    border:1px solid #6D28D9; border-radius:14px;
    padding:20px 24px; margin-bottom:16px;
}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_dark"
ACCENT          = "#38BDF8"

# ─────────────────────────────────────────────────────────────────────────────
#  MODULE-LEVEL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
# Regex that matches every common currency token and thousands-separator
_CURRENCY_RE       = r'(?:Rs\.?|PKR|USD|EUR|\$|£|€|,)'
# Minimum fraction of a column that must parse as numeric to be treated as
# a "dirty numeric" rather than a true categorical column
_DIRTY_THRESH_PCT  = 0.10


# ══════════════════════════════════════════════════════════════════════════════
#  CLASS 1 · DataPreprocessor
# ══════════════════════════════════════════════════════════════════════════════
class DataPreprocessor:
    """
    Handles all data-cleaning, imputation, and outlier-treatment logic.
    Always works on an internal copy so the original DataFrame is never mutated.
    """

    def __init__(self, df: pd.DataFrame):
        self.original = df.copy()
        self.df       = df.copy()

    # ── Profiling ─────────────────────────────────────────────────────────────
    def profile(self) -> pd.DataFrame:
        """Return a per-column quality summary."""
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

    # ── Currency / dirty-text → numeric (standalone helper) ───────────────────
    def convert_dirty_numerics(self) -> list[str]:
        """
        Strip currency tokens and coerce object columns to numeric when
        ≥ _DIRTY_THRESH_PCT of values parse successfully.
        Returns the list of converted column names.
        """
        converted = []
        threshold = max(1, len(self.df) * _DIRTY_THRESH_PCT)
        for col in self.df.columns:
            if self.df[col].dtype != 'object':
                continue
            cleaned = (
                self.df[col].astype(str)
                            .str.replace(_CURRENCY_RE, '', regex=True)
                            .str.strip()
            )
            temp = pd.to_numeric(cleaned, errors='coerce')
            if temp.notna().sum() >= threshold:
                self.df[col] = temp
                converted.append(col)
        return converted

    # ── Temporal column detection ─────────────────────────────────────────────
    def detect_temporal_columns(self) -> list[str]:
        """Return object columns that appear to store dates (≥ 80 % parse)."""
        candidates = []
        for col in self.df.select_dtypes(include='object').columns:
            sample = self.df[col].dropna().head(50)
            try:
                parsed = pd.to_datetime(sample, infer_datetime_format=True, errors='coerce')
                if parsed.notna().sum() / max(len(sample), 1) >= 0.80:
                    candidates.append(col)
            except Exception:
                pass
        return candidates

    # ── Sequential imputation ─────────────────────────────────────────────────
    def sequential_impute(self, method: str = "ffill") -> None:
        self.df = self.df.ffill() if method == "ffill" else self.df.bfill()

    # ── KNN imputation (numerical) ────────────────────────────────────────────
    def knn_impute(self, n_neighbors: int = 5) -> None:
        num_cols = self.df.select_dtypes(include=["int64", "float64"]).columns
        if num_cols.empty:
            return
        imp = KNNImputer(n_neighbors=n_neighbors, weights="distance")
        self.df[num_cols] = imp.fit_transform(self.df[num_cols])

    # ── Mode imputation (categorical) ─────────────────────────────────────────
    def mode_impute_categoricals(self) -> None:
        for col in self.df.select_dtypes(include='object').columns:
            mode = self.df[col].mode()
            self.df[col] = self.df[col].fillna(
                mode.iloc[0] if not mode.empty else "Unknown"
            )

    # ── Critical-column row drop ──────────────────────────────────────────────
    def drop_on_critical(self, critical_cols: list[str]) -> int:
        if not critical_cols:
            return 0
        before   = len(self.df)
        self.df  = self.df.dropna(subset=critical_cols)
        return before - len(self.df)

    # ── Outlier detection ─────────────────────────────────────────────────────
    def detect_outliers_iqr(self, col: str) -> pd.Series:
        Q1, Q3 = self.df[col].quantile([0.25, 0.75])
        IQR    = Q3 - Q1
        return (self.df[col] < Q1 - 1.5 * IQR) | (self.df[col] > Q3 + 1.5 * IQR)

    def detect_outliers_zscore(self, col: str, threshold: float = 3.0) -> pd.Series:
        z         = np.abs(zscore(self.df[col].dropna()))
        mask      = pd.Series(False, index=self.df.index)
        valid_idx = self.df[col].dropna().index
        mask[valid_idx] = z > threshold
        return mask

    def cap_outliers(self, col: str) -> None:
        """Winsorise: clip to [Q1 - 1.5·IQR,  Q3 + 1.5·IQR]."""
        Q1, Q3 = self.df[col].quantile([0.25, 0.75])
        IQR    = Q3 - Q1
        self.df[col] = self.df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

    def drop_outliers(self, mask: pd.Series) -> int:
        before   = len(self.df)
        self.df  = self.df[~mask]
        return before - len(self.df)

    # ══════════════════════════════════════════════════════════════════════════
    #  auto_clean  ·  The bulletproof pipeline
    # ══════════════════════════════════════════════════════════════════════════
    def auto_clean(self, k_neighbors: int = 5) -> list[str]:
        """
        All-in-one auto-clean pipeline in strict, dependency-correct order.

        Steps
        ─────
        1  Duplicate removal
        2  Text normalisation   – .title() on TRUE categoricals only
        3  Numeric conversion   – strip currency tokens → pd.to_numeric (coerce)
        4  Bulletproof ghost fix – mask ALL negative numerics → NaN
        5  KNN imputation + categorical mode fill
        6  IQR Winsorisation

        Why a mathematical mask instead of df.replace(-999)?
        ─────────────────────────────────────────────────────
        After string → float conversion, '-999' becomes -999.0 (float64).
        df.replace(-999, np.nan) compares an int against a float; edge-case
        binary representations can cause silent misses.  A `< 0` comparison
        operates on the numeric VALUE, making it completely immune to
        floating-point precision artefacts.  Prices, salaries, ages, and
        quantities are never legitimately negative, so any negative number
        is definitionally a garbage sentinel.

        Parameters
        ──────────
        k_neighbors : int  – neighbours for KNNImputer (default 5)

        Returns
        ───────
        list[str]  – human-readable log of every action performed
        """
        log: list[str] = []

        # ── Step 1 · Drop duplicates ─────────────────────────────────────────
        n_dupes = self.drop_duplicates()
        if n_dupes:
            log.append(f"✅ Removed {n_dupes} duplicate row(s).")

        # ── Column classification  (happens BEFORE any data is modified) ──────
        #
        #   Probe every object column:
        #     • If ≥ 10 % of values parse as numbers after symbol-stripping
        #       → dirty_numeric  (e.g. 'Rs. 500', '$1,200 PKR')
        #     • Otherwise → true_categorical  (e.g. 'Karachi', 'Male')
        #
        #   This ensures .title() is NEVER applied to price / qty columns, and
        #   numeric coercion is NEVER applied to legitimate text columns.
        #
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
                pass  # guard against unexpected column states

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

        # ── Step 4 · Bulletproof ghost-value fix ─────────────────────────────
        #
        #   Select ALL numeric columns (includes columns that were already
        #   numeric before cleaning started, not just the ones we just
        #   converted).  Mask every value < 0 to NaN in a single vectorised
        #   operation — no loops, no .replace(), no float-precision risk.
        #
        num_cols = self.df.select_dtypes(include='number').columns
        if not num_cols.empty:
            neg_mask  = self.df[num_cols] < 0
            neg_count = int(neg_mask.sum().sum())
            self.df[num_cols] = self.df[num_cols].mask(neg_mask, np.nan)
            log.append(
                f"✅ Bulletproof ghost fix: {neg_count} negative value(s) "
                "masked → NaN (mathematical comparison, float-precision safe)."
            )
        else:
            log.append("ℹ️  No numeric columns found for ghost-value masking.")

        # ── Step 5 · Algorithmic imputation ──────────────────────────────────
        missing_before = int(self.df.isnull().sum().sum())

        #   5a · KNN on numeric columns  (fills ghost-NaNs with smart estimates)
        num_cols = self.df.select_dtypes(include='number').columns.tolist()
        if num_cols:
            try:
                imp               = KNNImputer(n_neighbors=k_neighbors, weights='distance')
                self.df[num_cols] = imp.fit_transform(self.df[num_cols])
            except ValueError as exc:
                log.append(f"⚠️  KNN imputation failed: {exc}")

        #   5b · Mode fill for remaining categorical columns
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
        capped: list[str] = []
        for col in self.df.select_dtypes(include='number').columns:
            try:
                self.cap_outliers(col)
                capped.append(col)
            except (TypeError, ValueError):
                pass  # guard against all-NaN edge-case columns

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
        """Extended describe() enriched with skewness and kurtosis."""
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
#  CLASS 4 · UIBuilder
# ══════════════════════════════════════════════════════════════════════════════
class UIBuilder:
    """
    Owns every Streamlit call.
    Delegates all computation to the other three classes.
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

        # ── Mode selector ─────────────────────────────────────────────────────
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
                '<h4 style="color:#A78BFA;margin-top:0">🤖 Auto-Clean — Bulletproof Pipeline</h4>'
                '<p style="color:#C4B5FD;margin:0">Runs six battle-tested steps in strict dependency order. '
                'Ghost values (-999, -9999, -1000…) are destroyed via a <b>mathematical mask</b> '
                '(<code>value &lt; 0 → NaN</code>) that is immune to float-precision mismatches.</p>'
                "</div>",
                unsafe_allow_html=True,
            )

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

        # FIX: Agar cleaned_df None hai, toh sahi se raw_df par fallback karein
        cleaned_df = st.session_state.get("cleaned_df")
        df = cleaned_df if cleaned_df is not None else self.raw_df
        
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

        # Before / after histogram
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
                         marker_color="#86EFAC", opacity=0.75, nbinsx=40),
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

        # FIX: Sahi se check karein ki cleaned_df available hai ya nahi
        cleaned_df = st.session_state.get("cleaned_df")
        df = cleaned_df.copy() if cleaned_df is not None else self.raw_df.copy()
        
        fe  = FeatureEngineer(df)

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
    #  TAB 5 · Advanced EDA
    # ─────────────────────────────────────────────────────────────────────────
    def render_eda_tab(self):
        st.header("📈 Advanced Exploratory Data Analysis")

        # FIX: Hierarchy ke mutabik check karein jo pehle mile aur None na ho
        if st.session_state.get("cleaned_df") is not None:
            df_source = st.session_state["cleaned_df"]
            source_label = "Cleaned Data"
        elif st.session_state.get("engineered_df") is not None:
            df_source = st.session_state["engineered_df"]
            source_label = "Engineered Data"
        else:
            df_source = self.raw_df
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
                sig_rows = []
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

        # ── E · PCA ───────────────────────────────────────────────────────────
        st.markdown("### 🔮 E · Principal Component Analysis (PCA)")
        n_pca = st.slider(
            "Number of Principal Components", 2, min(10, len(analyzer.num_cols)), 3
        )

        if len(analyzer.num_cols) < 2:
            st.info("Need ≥ 2 numerical columns for PCA.")
        else:
            try:
                scores, evr, pca_obj = analyzer.run_pca(n_components=n_pca)

                # Scree plot
                fig_scree = go.Figure(go.Bar(
                    x=[f"PC{i+1}" for i in range(len(evr))],
                    y=(evr * 100).round(2),
                    marker_color=ACCENT,
                    text=(evr * 100).round(1),
                    textposition="outside",
                ))
                fig_scree.update_layout(
                    title="Scree Plot — Explained Variance per Component",
                    yaxis_title="Explained Variance (%)",
                    template=PLOTLY_TEMPLATE, height=350,
                )
                st.plotly_chart(fig_scree, use_container_width=True)

                # 2-D biplot
                color_col = None
                if analyzer.cat_cols:
                    c_choice = st.selectbox(
                        "Colour PCA by (optional)", ["None"] + analyzer.cat_cols,
                        key="pca_color",
                    )
                    if c_choice != "None":
                        valid_idx = df_source[analyzer.num_cols].dropna().index
                        scores[c_choice] = df_source.loc[valid_idx, c_choice].values
                        color_col = c_choice

                fig_2d = px.scatter(
                    scores, x="PC1", y="PC2", color=color_col,
                    title=(
                        f"PCA 2D — "
                        f"{round(evr[0]*100,1)}% + {round(evr[1]*100,1)}% "
                        "explained variance"
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
                        columns=[f"PC{i+1}" for i in range(n_pca)],
                    ).round(4)
                    st.dataframe(loadings, use_container_width=True)

            except Exception as exc:
                st.error(f"PCA failed: {exc}")

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
            """
        )
        st.sidebar.markdown("---")
        uploaded = st.sidebar.file_uploader("📂 Upload CSV Dataset", type=["csv"])
        if uploaded:
            st.sidebar.success("Dataset loaded!")
        st.sidebar.markdown("---")
        st.sidebar.info(
            "**Tip:** Use Auto-Clean for datasets with mixed currency "
            "strings or sentinel ghost values (-999, -9999, …)."
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
    st.caption("Production-grade data preprocessing and exploratory analysis dashboard.")

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
        st.session_state["last_file"]     = uploaded_file.name
        st.session_state["cleaned_df"]    = None
        st.session_state["engineered_df"] = None

    ui = UIBuilder(raw_df)

    tabs = st.tabs([
        "📋 Raw Data",
        "⚙️ Preprocessing",
        "🎯 Outlier Analysis",
        "🔧 Feature Engineering",
        "📈 Advanced EDA",
    ])

    with tabs[0]: ui.render_raw_tab()
    with tabs[1]: ui.render_preprocessing_tab()
    with tabs[2]: ui.render_outlier_tab()
    with tabs[3]: ui.render_feature_tab()
    with tabs[4]: ui.render_eda_tab()


if __name__ == "__main__":
    main()