# AutoClean AI

AutoClean AI is an end-to-end Data Preprocessing and Dataset Cleaning web application built with **Streamlit**. Upload a raw **CSV** or **Excel (XLSX)** file, and the app will automatically profile the dataset, handle missing values, remove duplicates, correct data types, detect and treat outliers, encode categorical features, scale numeric features, visualize results, generate a dataset health report, and export a cleaned, ML-ready dataset.

## Features

- **Dataset Upload** (CSV/XLSX)
- **Profiling Dashboard**
  - Row/column counts
  - Missing values
  - Duplicate row count
  - Numerical vs categorical columns
  - Memory usage
  - Data type summary
- **Missing Value Handling**
  - Mean / Median / Mode imputation
  - KNN imputation for numeric data
- **Duplicate Removal**
- **Automatic Data Type Correction**
  - Convert numeric strings to numeric
  - Parse datetime-like columns
- **Outlier Detection (IQR)**
  - Remove outliers
  - Cap (winsorize-style) outliers
- **Categorical Encoding**
  - Label Encoding
  - One-Hot Encoding
- **Feature Scaling**
  - StandardScaler
  - MinMaxScaler
- **Interactive Visualizations (Plotly)**
  - Missing values bar chart
  - Correlation heatmap
  - Boxplots
  - Histograms
- **Dataset Health Score (0-100)**
  - Before cleaning vs after cleaning
- **Download Cleaned Dataset**

## Screenshots

> Add screenshots here once you run the app.

## Folder Structure

```
AutoCleanAI/
├── app.py
├── preprocessing.py
├── visualization.py
├── utils.py
├── requirements.txt
├── README.md
├── datasets/
└── outputs/
```

## Installation

1. Open a terminal in the project directory.
2. Create a virtual environment (recommended):

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit app:

```bash
streamlit run app.py
```

2. Upload a CSV/XLSX file.
3. Configure cleaning options in the sidebar.
4. Review profiling + visualizations.
5. Download the cleaned dataset.

## Future Improvements

- Add support for more imputation strategies (e.g., iterative imputer)
- Add advanced date feature engineering
- Add automated ML preprocessing pipelines per task type
- Add model training preview (e.g., baseline classifiers/regressors)
- Add better handling for mixed-type columns and text cleaning

