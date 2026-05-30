# ezclean-data

[![PyPI Version](https://img.shields.io/pypi/v/ezclean-data.svg)](https://pypi.org/project/ezclean-data/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/ezclean-data.svg)](https://pypi.org/project/ezclean-data/)

A comprehensive, dataset-agnostic Python library for automated data ingestion, cleaning, preprocessing, and exploratory data analysis.

`ezclean-data` streamlines the repetitive tasks involved in preparing datasets by providing intelligent loading, automated sanitization, statistical summaries, advanced visualizations, and standalone interactive dashboards.

---

## Overview

Data preparation often consumes a significant portion of the data science workflow. `ezclean-data` provides a unified interface that automatically loads structured datasets, cleans inconsistencies, handles missing values and outliers, standardizes column names, and generates insightful visualizations with minimal code.

The library is designed to work across a wide variety of dataset formats and structures, making it suitable for students, researchers, analysts, and machine learning practitioners.

---

## Features

### Smart Data Loading

- Automatically detects file formats and selects the appropriate Pandas engine.
- Supports both local files and remote URLs.
- Handles a broad range of structured data formats.

### Automated Data Cleaning

- Standardizes column names into consistent `snake_case` format.
- Detects and replaces common invalid placeholder values.
- Performs automatic type correction where appropriate.
- Handles missing values using intelligent, data-type-aware strategies.
- Detects and mitigates outliers using the Interquartile Range (IQR) method.

### Exploratory Data Analysis

- Generates detailed column summaries and completeness statistics.
- Provides automatic visualizations based on column data types.
- Creates generalized pairplot-style relationship matrices for rapid exploration.

### Interactive Dashboard Generation

- Produces self-contained HTML dashboards.
- Includes summary statistics and data quality metrics.
- Provides interactive Plotly-based visualizations.
- Works entirely offline once generated.

---

## Installation

Install directly from PyPI:

```bash
pip install ezclean-data
```

---

## Quick Start

```python
from ezclean import Smart_loader, Cleaner, colname, plot, plot_dashboard

# Load dataset
df = Smart_loader("tested.csv")

# Execute cleaning pipeline
df_cleaned = Cleaner(df)

# Display column statistics
colname(df_cleaned)

# Visualize a single column
plot(df_cleaned, "survived")

# Generate a relationship matrix
plot(df_cleaned)

# Create an interactive dashboard
plot_dashboard(
    df_cleaned,
    filename="my_dashboard.html"
)
```

---

# API Reference

## Smart_loader()

```python
Smart_loader(file_path, **kwargs)
```

Automatically loads structured datasets from local storage or remote URLs.

### Supported Formats

| Category | Formats |
|-----------|----------|
| Text Files | CSV, TSV, TXT |
| JSON Formats | JSON, JSONL, NDJSON |
| Spreadsheet Files | XLSX, XLS, ODS |
| Columnar Formats | Parquet, Feather, Arrow, ORC |
| Statistical Formats | SPSS, SAS, Stata |
| Other Formats | XML, HTML, Pickle, HDF |

---

## Cleaner()

```python
Cleaner(df, ...)
```

Executes a complete data-cleaning pipeline.

### Included Operations

#### Column Name Standardization

- Converts names to `snake_case`
- Removes special characters
- Eliminates duplicate separators

#### Data Sanitization

Replaces common placeholder values such as:

```text
?
NULL
null
nil
N/A
NaN
```

with proper missing-value representations.

#### Text Normalization

- Trims whitespace
- Standardizes string formatting

#### Automatic Type Detection

- Converts columns to numeric types when appropriate
- Preserves incompatible values

#### Missing Value Handling

- Numerical columns → Median Imputation
- Categorical columns → `"Unknown"` Replacement

#### Outlier Treatment

- Uses Interquartile Range (IQR) thresholds
- Removes extreme observations automatically

---

## colname()

```python
colname(df)
```

Displays detailed metadata for each column, including:

- Data type
- Missing value count
- Completeness percentage
- Unique value count
- Statistical summaries

---

## plot()

```python
plot(df, target_column=None, columns=None)
```

### Single-Column Visualization

When a target column is specified, the visualization is selected automatically based on data type.

| Data Type | Visualization |
|------------|---------------|
| Numeric | Histogram + Box Plot |
| Categorical | Bar Chart + Donut Chart |
| Datetime | Trend Line |

### Relationship Matrix

```python
plot(df)
```

Generates a generalized pairplot matrix displaying:

- Univariate distributions
- Correlation patterns
- Relationships between variables

---

## plot_dashboard()

```python
plot_dashboard(
    df,
    filename="ezclean_dashboard.html",
    show=True
)
```

Creates a standalone interactive dashboard containing:

### Dataset Summary

- Dataset dimensions
- Completeness metrics
- Data quality statistics

### Column Analysis

- Data types
- Missing values
- Unique value counts

### Interactive Visualization Builder

Users can dynamically select:

- X-axis variables
- Y-axis variables
- Plot types

without writing additional code.

### Relationship Matrix

Embedded Plotly-based exploratory visualization for multivariate analysis.

---

## Example Workflow

```python
from ezclean import *

df = Smart_loader("data.csv")

df = Cleaner(df)

colname(df)

plot(df, "age")

plot(df)

plot_dashboard(
    df,
    filename="dashboard.html"
)
```

---

## Use Cases

- Data Science Projects
- Machine Learning Preprocessing
- Academic Research
- Exploratory Data Analysis
- Business Intelligence Reporting
- Rapid Dataset Validation
- Educational Applications

---

## Why ezclean-data?

Most data analysis projects begin with repetitive preprocessing tasks such as loading files, cleaning columns, handling missing values, detecting outliers, and creating exploratory visualizations.

`ezclean-data` consolidates these operations into a simple and consistent workflow, allowing users to focus on analysis and model development rather than boilerplate data preparation code.

---

## License

This project is licensed under the MIT License.

See the `LICENSE` file for complete licensing information.

---

## Author

Developed and maintained by **Thilac Ramesh**.

Contributions, feature requests, and issue reports are welcome.
