"""
Project: AutoClean & EDA Toolkit
Course: Introduction to Data Science (IDS)
Description: A Streamlit dashboard for automated dataset cleaning and exploratory data analysis.
Dependencies: streamlit, pandas, plotly, statsmodels
"""

import streamlit as st
import pandas as pd
import plotly.express as px

# Configure main page settings
st.set_page_config(
    page_title="AutoClean & EDA Toolkit",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a clean, modern dark UI
st.markdown(
    """
    <style>
        .stApp { background-color: #0E1117; color: #FAFAFA; }
        div[data-testid="stMetric"] {
            background-color: #1B1F27;
            border: 1px solid #2D333B;
            border-radius: 10px;
            padding: 15px;
        }
        button[data-baseweb="tab"] { font-size: 16px; font-weight: 600; }
        h1, h2, h3 { color: #4FD1C5; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar setup for file ingestion
st.sidebar.title("🧹 AutoClean & EDA Toolkit")
st.sidebar.markdown("Upload a raw CSV dataset to clean and analyze it interactively.")

uploaded_file = st.sidebar.file_uploader("📂 Upload CSV file", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.info("**Pipeline:**\n1. Preview Data\n2. Auto-Clean\n3. EDA Visualizations")

st.title("🧹 AutoClean & EDA Toolkit")
st.caption("End-to-end automated data preprocessing and visualization dashboard.")

# Execute main logic only if a dataset is provided
if uploaded_file is not None:

    # Load dataset safely
    try:
        raw_df = pd.read_csv(uploaded_file)
        st.sidebar.success("File ingested successfully!")
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

    # Initialize session state to store the cleaned dataframe memory
    if "cleaned_df" not in st.session_state:
        st.session_state.cleaned_df = None

    # App layout divided into 3 functional tabs
    tab1, tab2, tab3 = st.tabs(["📋 Raw Data", "⚙️ Preprocessing", "📈 EDA"])

    # ------------------------------------------------------------------
    # Tab 1: Data Preview & Profiling
    # ------------------------------------------------------------------
    with tab1:
        st.header("📋 Raw Data Preview")

        # Display dataset dimensions
        col1, col2 = st.columns(2)
        col1.metric("Total Rows", f"{raw_df.shape[0]:,}")
        col2.metric("Total Columns", f"{raw_df.shape[1]:,}")

        st.markdown("### Top 10 Records")
        st.dataframe(raw_df.head(10), use_container_width=True)

        st.markdown("### Data Types & Missing Values Profile")
        try:
            # Generate a summary dataframe for initial profiling
            profile_df = pd.DataFrame({
                "Feature": raw_df.columns,
                "Dtype": raw_df.dtypes.astype(str).values,
                "Null Count": raw_df.isnull().sum().values,
                "Null (%)": (raw_df.isnull().mean() * 100).round(2).values,
            })
            st.dataframe(profile_df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Profiling error: {e}")

    # ------------------------------------------------------------------
    # Tab 2: Automated Cleaning Pipeline
    # ------------------------------------------------------------------
    with tab2:
        st.header("⚙️ Automated Preprocessing")

        missing_before = int(raw_df.isnull().sum().sum())
        duplicates_before = int(raw_df.duplicated().sum())

        st.markdown("### Quality Report (Pre-Clean)")
        col1, col2 = st.columns(2)
        col1.metric("Total Missing Values", f"{missing_before:,}")
        col2.metric("Duplicate Rows", f"{duplicates_before:,}")

        st.markdown("---")

        # --- SMART UI: Selective Dropping ---
        st.write("### 🛠️ Cleaning Settings")
        st.info(
            "Select the most important columns (e.g., Price, Total). If data is "
            "missing or corrupted here, the entire row will be dropped. For all "
            "other columns, Median/Mode will be used to save data."
        )

        critical_cols = st.multiselect(
            "Select Critical Columns:",
            options=raw_df.columns.tolist()
        )
        # ------------------------------------

        # Trigger cleaning process
        if st.button("🧼 Execute Auto-Clean", type="primary"):
            try:
                cleaned = raw_df.copy()

                # Remove identical records
                cleaned = cleaned.drop_duplicates()

                # --- BULLETPROOF LOGIC: Aggressive Numeric Conversion ---
                # Detects dirty text in numeric columns and forces them to numbers.
                # FIX: Use a precise regex so we only strip currency symbols and
                # thousands separators ($ , Rs PKR) instead of stray letters
                # like a lone "R" or "s" that could corrupt real text columns.
                currency_pattern = r'(Rs\.?|PKR|USD|\$|,)'

                for col in cleaned.columns:
                    if cleaned[col].dtype == 'object':
                        clean_str = cleaned[col].astype(str).str.replace(
                            currency_pattern, '', regex=True
                        ).str.strip()

                        temp_num = pd.to_numeric(clean_str, errors='coerce')

                        # Apply conversion only if at least 10% of the data
                        # successfully converts to a number AND the column
                        # is not empty.
                        if len(cleaned[col]) > 0 and (
                            temp_num.notna().sum() >= (len(cleaned[col]) * 0.1)
                        ):
                            cleaned[col] = temp_num
                # --------------------------------------------------------

                # --- IMPLEMENTING SELECTIVE DROPPING LOGIC ---
                # 1. Drop rows ONLY if the 'Critical Columns' have missing/garbage values
                if critical_cols:
                    cleaned = cleaned.dropna(subset=critical_cols)

                # FIX: Guard against an empty dataframe after dropna(), which
                # would break median()/mode() calculations and the EDA tab.
                if cleaned.empty:
                    st.warning(
                        "⚠️ After dropping rows with missing values in the selected "
                        "critical columns, no data remains. Please choose different "
                        "critical columns or skip selective dropping."
                    )
                    st.session_state.cleaned_df = None
                else:
                    # Separate feature types for targeted imputation on REMAINING data
                    num_cols = cleaned.select_dtypes(include=["int64", "float64"]).columns
                    cat_cols = cleaned.select_dtypes(include=["object"]).columns

                    # 2. Safely Fill Median for the rest of the numbers
                    for col in num_cols:
                        median_val = cleaned[col].median()
                        # If median itself is NaN (all values missing), fall back to 0
                        if pd.isna(median_val):
                            median_val = 0
                        cleaned[col] = cleaned[col].fillna(median_val)

                    # 3. Safely Fill Mode for the text data
                    for col in cat_cols:
                        mode_val = cleaned[col].mode()
                        fill_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
                        cleaned[col] = cleaned[col].fillna(fill_val)
                    # ---------------------------------------------

                    # Save processed data to state
                    st.session_state.cleaned_df = cleaned
                    st.success("✅ Preprocessing completed successfully!")

            except Exception as e:
                st.error(f"Cleaning execution failed: {e}")

        # ------------------------------------------------------------
        # Display results post-cleaning + DOWNLOAD CLEANED CSV
        # ------------------------------------------------------------
        if st.session_state.cleaned_df is not None:
            final_df = st.session_state.cleaned_df

            st.markdown("### Quality Report (Post-Clean)")
            col1, col2 = st.columns(2)
            col1.metric("Total Missing Values", int(final_df.isnull().sum().sum()))
            col2.metric("Duplicate Rows", int(final_df.duplicated().sum()))

            st.markdown("### 🔍 Cleaned Data Preview")
            st.dataframe(final_df.head(10), use_container_width=True)

            st.markdown("---")
            st.markdown("### ⬇️ Download Your Cleaned Dataset")

            # Convert cleaned dataframe to CSV bytes for download
            try:
                csv_data = final_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Cleaned Data (CSV)",
                    data=csv_data,
                    file_name="cleaned_data.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True,
                    key="download_cleaned_csv",
                )
            except Exception as e:
                st.error(f"⚠️ Could not prepare the cleaned CSV for download: {e}")
        else:
            st.info("ℹ️ Click '🧼 Execute Auto-Clean' above to clean the dataset and enable the download option.")

    # ------------------------------------------------------------------
    # Tab 3: Exploratory Data Analysis
    # ------------------------------------------------------------------
    with tab3:
        st.header("📈 Exploratory Data Analysis")

        # Prefer cleaned data for EDA if available
        if st.session_state.cleaned_df is not None:
            eda_data = st.session_state.cleaned_df
            st.caption("Status: Using Cleaned Data")
        else:
            eda_data = raw_df
            st.caption("Status: Using Raw Data (Run preprocessing for better results)")

        try:
            num_features = eda_data.select_dtypes(include=["int64", "float64"]).columns.tolist()

            if not num_features:
                st.warning("Insufficient numerical features for plotting.")
            else:
                # 1. Bivariate Analysis: Scatter Plot with Regression Line
                st.markdown("### 🔵 Feature Relationship (Scatter)")
                col_x, col_y = st.columns(2)
                x_feat = col_x.selectbox("X-axis feature", num_features, key="x")
                y_feat = col_y.selectbox("Y-axis feature", num_features, index=min(1, len(num_features) - 1), key="y")

                try:
                    if x_feat == y_feat:
                        st.warning("Please select different features for X and Y axes to generate a scatter plot.")
                    else:
                        fig_scatter = px.scatter(
                            eda_data, x=x_feat, y=y_feat, trendline="ols",
                            title=f"OLS Trendline: {y_feat} vs {x_feat}",
                            template="plotly_dark", opacity=0.7
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
                except Exception:
                    st.error("Scatter plot failed. Ensure 'statsmodels' is installed for the trendline.")

                st.markdown("---")

                # 2. Univariate Analysis: Distribution Plot
                st.markdown("### 🟢 Distribution Analysis")
                dist_feat = st.selectbox("Select feature for distribution", num_features)

                fig_hist = px.histogram(
                    eda_data, x=dist_feat, nbins=30, marginal="box",
                    title=f"Distribution & Spread of {dist_feat}",
                    template="plotly_dark", color_discrete_sequence=["#4FD1C5"]
                )
                st.plotly_chart(fig_hist, use_container_width=True)

                st.markdown("---")

                # 3. Multivariate Analysis: Correlation Matrix
                st.markdown("### 🟣 Feature Correlation Heatmap")
                if len(num_features) >= 2:
                    corr_matrix = eda_data[num_features].corr()
                    fig_heat = px.imshow(
                        corr_matrix, text_auto=".2f", aspect="auto",
                        color_continuous_scale="RdBu_r", template="plotly_dark"
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)
                else:
                    st.info("Need at least 2 numerical columns to compute a correlation heatmap.")

        except Exception as e:
            st.error(f"Visualization engine encountered an error: {e}")

else:
    st.info("Awaiting dataset. Please upload a CSV file via the sidebar.")