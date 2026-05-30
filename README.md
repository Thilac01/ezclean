# ✨ ezclean-data

[![PyPI Version](https://img.shields.io/pypi/v/ezclean-data.svg)](https://pypi.org/project/ezclean-data/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python versions](https://img.shields.io/pypi/pyversions/ezclean-data.svg)](https://pypi.org/project/ezclean-data/)

A premium, dataset-agnostic Python library designed to automate the painful parts of data loading, cleaning, and exploration. With `ezclean-data`, you can load any structured file format, sanitize outliers and null values, and instantly produce beautiful, interactive visualization dashboards or multi-variable pairplot matrices.

---

## 🚀 Features

- 📥 **Smart Data Loader**: Auto-detects extensions (CSV, Excel, Parquet, JSON, etc.) and routes them to optimized Pandas engines, streaming directly over HTTP or loading local paths.
- 🧼 **Intelligent Data Cleaner**: Standardizes columns to `snake_case`, handles structural garbage strings, handles outliers via IQR boundary thresholds, and fills null values using type-specific heuristics (e.g. median for numbers).
- 📊 **Universal Plot Grid**: Renders a generalized interactive Plotly pairplot matrix of subplots showing all possible univariate distributions (diagonal) and bivariate relationships (off-diagonal) for any dataset.
- 🎨 **Standalone HTML Dashboard**: Generates a fully interactive, lightweight dashboard with statistics cards, a column definitions table, and a dynamic JavaScript plot builder that works fully offline!

---

## 📦 Installation

Install `ezclean-data` directly from PyPI:

```bash
pip install ezclean-data
```

---

## ⚡ Quick Start

```python
from ezclean import Smart_loader, Cleaner, colname, plot, plot_dashboard

# 1. Load your dataset from a file or url
df = Smart_loader("tested.csv")

# 2. Run the unified cleaning pipeline
df_cleaned = Cleaner(df)

# 3. Print column statistics
colname(df_cleaned)

# 4. Plot a single column (auto-detects types)
plot(df_cleaned, "survived")

# 5. Plot the generalized pairplot matrix (all combinations)
plot(df_cleaned)

# 6. Generate and open a gorgeous standalone HTML Dashboard
plot_dashboard(df_cleaned, filename="my_dashboard.html")
```

---

## 🛠️ Module API Overview

### 1. `Smart_loader(file_path, **kwargs)`
Instantly routes local or remote URLs to Pandas readers. Supported formats:
`csv`, `tsv`, `txt`, `json`, `jsonl`, `ndjson`, `excel` (`xlsx`, `xls`, `ods`), `parquet`, `feather`, `arrow`, `orc`, `xml`, `html`, `pickle`, `stata`, `spss`, `sas`, `hdf`.

### 2. `Cleaner(df, ...)`
High-level cleaning pipeline wrapping:
- `column_name_sanity()`: Clean symbols, Deduplicate underscores, Convert to `snake_case`.
- `sanitize_data()`: Replaces structural garbage (`?`, `NULL`, `nil`) with NumPy NaNs.
- `text_normalization()`: Trims whitespace and normalizes string fields.
- `auto_type_correction()`: Converts column dtypes to numeric if >50% of values match.
- `intelligent_null_filling()`: Median imputes numeric fields; fills categorical values with `"Unknown"`.
- `handle_outliers()`: IQR-based outlier trimming.

### 3. `plot(df, target_column=None, columns=None)`
- If `target_column` is provided, renders a single visual (numerical gets Box+Histogram; categorical gets Donut+Bar; datetime gets Line Trend).
- If `target_column=None`, renders `plot_matrix` containing all univariate and bivariate subplots for selected columns (default: top 5).

### 4. `plot_dashboard(df, filename="ezclean_dashboard.html", show=True)`
Writes a self-contained, interactive HTML dashboard containing:
1. Complete column completeness summary tables.
2. Dynamic Plotly client visualizer where users can build custom X vs Y plots.
3. Embedded pairplot relation matrix.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
