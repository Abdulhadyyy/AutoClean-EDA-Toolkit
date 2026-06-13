# 🧹 AutoClean & EDA Toolkit

An automated data preprocessing and visualization dashboard built with Python and Streamlit. This project was developed as part of the **Introduction to Data Science (IDS)** coursework for the BS Data Science program to demonstrate practical handling of messy, real-world datasets.

## 🚀 Overview
Real-world data is rarely clean. This application provides a seamless, interactive pipeline to ingest raw CSV files, intelligently identify and resolve structural anomalies (like garbage text in numeric columns), and perform comprehensive Exploratory Data Analysis (EDA) without writing boilerplate code.

## ✨ Key Features
- **Smart Type Casting:** Automatically detects and strips currency symbols ($, Rs, PKR) and commas from numerical columns, converting corrupted object types back to strictly numeric (`float64`) types.
- **Selective Dropping Logic:** Allows users to define "Critical Columns" (e.g., Price, Total). Completely corrupted rows in these columns are permanently dropped to maintain strict data integrity.
- **Automated Imputation:** Missing values in remaining numerical columns are imputed using the **Median** (to prevent variance distortion), while categorical columns use the **Mode**.
- **Interactive EDA Visualization:** Leverages Plotly to generate:
  - Bivariate Analysis (Scatter plots with OLS regression trendlines).
  - Univariate Analysis (Histograms with marginal box plots for distribution spread).
  - Multivariate Analysis (Pearson Correlation Heatmaps).
- **Clean Data Export:** One-click download of the thoroughly preprocessed dataset as a CSV file.

## 🛠️ Tech Stack
- **Language:** Python 3
- **Framework:** Streamlit (Web UI)
- **Data Manipulation:** Pandas, NumPy
- **Data Visualization:** Plotly Express
- **Statistical Modeling:** Statsmodels (for OLS regression)

## 💻 Installation & Usage (Local Development)

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Abdulhadyyy/AutoClean-EDA-Toolkit.git](https://github.com/Abdulhadyyy/AutoClean-EDA-Toolkit.git)
   cd AutoClean-EDA-Toolkit
