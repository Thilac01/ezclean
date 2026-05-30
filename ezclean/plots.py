import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import webbrowser
import os
import tempfile
import plotly.io as pio

def colname(df: pd.DataFrame):
    """
    Prints a clean, structured table of all column names, their data types,
    and missing values count so you can easily choose which column to plot.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Please provide a valid pandas DataFrame.")
        return
        
    print("\n" + "="*60)
    print(f"{'COLUMN NAME':<30} | {'DATA TYPE':<15} | {'MISSING NULLS':<10}")
    print("="*60)
    
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        null_count = df[col].isna().sum()
        print(f"{col:<30} | {dtype_str:<15} | {null_count:<10}")
    print("="*60 + "\n")


def _classify_column(col_data: pd.Series, col_name: str) -> str:
    """
    Classifies a pandas series into: 'datetime', 'numeric', or 'categorical'.
    """
    clean_data = col_data.dropna()
    if len(clean_data) == 0:
        return 'categorical'
        
    # Check datetime types
    if pd.api.types.is_datetime64_any_dtype(col_data):
        return 'datetime'
    
    # Try converting a sample to datetime if string and named like date/time
    if col_data.dtype == 'object' or str(col_data.dtype).startswith('str'):
        if any(kw in str(col_name).lower() for kw in ['date', 'time', 'timestamp', 'year', 'month', 'day']):
            try:
                # Try parsing first 100 rows
                pd.to_datetime(clean_data.head(100), errors='raise')
                return 'datetime'
            except Exception:
                pass

    # Check numeric types
    if pd.api.types.is_numeric_dtype(col_data):
        # Float dtypes or high cardinality integer are numeric
        if pd.api.types.is_float_dtype(col_data):
            return 'numeric'
        if clean_data.nunique() > 10:
            return 'numeric'
        return 'categorical'
        
    return 'categorical'


def plot_single_column(df: pd.DataFrame, target_column: str):
    """
    Intelligently analyzes a single column's data type and renders
    its best matching optimized chart setup instantly.
    """
    if target_column not in df.columns:
        print(f"Error: Column '{target_column}' does not exist in the DataFrame.")
        return

    col_data = df[target_column]
    col_type = _classify_column(col_data, target_column)

    # Sample if the dataframe is very large to make rendering instantaneous
    plot_df = df.sample(n=min(5000, len(df)), random_state=42) if len(df) > 5000 else df
    col_plot_data = plot_df[target_column]
    
    # Clean data for plotting
    non_null_data = col_plot_data.dropna()
    if len(non_null_data) == 0:
        print(f"Warning: Column '{target_column}' has all missing values. Cannot generate plot.")
        return

    # 1. NUMERICAL -> Histogram + Box Plot Subplot
    if col_type == 'numeric':
        fig = make_subplots(rows=1, cols=2, subplot_titles=(f"{target_column} Box Plot", f"{target_column} Histogram"))
        
        # Teal / Slate theme
        fig.add_trace(go.Box(x=non_null_data, name=target_column, boxpoints='outliers', marker_color='#26A69A'), row=1, col=1)
        fig.add_trace(go.Histogram(x=non_null_data, name=target_column, nbinsx=30, marker_color='#455A64', marker_line=dict(color='white', width=0.5)), row=1, col=2)
        
        fig.update_layout(
            height=450, 
            title_text=f"<b>Distribution Analysis: {target_column}</b>", 
            showlegend=False, 
            template="plotly_white",
            font=dict(family="Inter, sans-serif")
        )
        fig.show()

    # 2. DATETIME -> Timeline Tracking Plot
    elif col_type == 'datetime':
        temp_df = plot_df.dropna(subset=[target_column]).copy()
        temp_df[target_column] = pd.to_datetime(temp_df[target_column], errors='coerce')
        temp_df = temp_df.dropna(subset=[target_column])
        
        if len(temp_df) == 0:
            print(f"Warning: Could not parse dates in '{target_column}'.")
            return
            
        temp_df['year_month'] = temp_df[target_column].dt.to_period('M').astype(str)
        
        # See if there is a numeric column we can plot a trend for
        num_cols = [c for c in df.columns if _classify_column(df[c], c) == 'numeric' and c != target_column]
        metric_col = num_cols[0] if num_cols else None
        
        if metric_col:
            timeline_df = temp_df.groupby('year_month')[metric_col].mean().reset_index().sort_values('year_month')
            fig = px.line(
                timeline_df, x='year_month', y=metric_col, 
                title=f"<b>Timeline Trend:</b> Average {metric_col} over time ({target_column})",
                template="plotly_white",
                markers=True
            )
            fig.update_traces(line=dict(color='#AB47BC', width=3), marker=dict(size=8))
        else:
            timeline_df = temp_df.groupby('year_month').size().reset_index(name='count').sort_values('year_month')
            fig = px.line(
                timeline_df, x='year_month', y='count', 
                title=f"<b>Timeline Volume:</b> Record Volume over time ({target_column})",
                template="plotly_white",
                markers=True
            )
            fig.update_traces(line=dict(color='#8E24AA', width=3), marker=dict(size=8))

        fig.update_layout(
            height=450,
            font=dict(family="Inter, sans-serif")
        )
        fig.show()

    # 3. CATEGORICAL -> Composition Pie + Bar Subplot
    else:
        # Get top 10 categories to avoid clutter
        value_counts = non_null_data.value_counts().head(10)
        
        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "domain"}, {"type": "xy"}]],
                            subplot_titles=("Category Distribution (Donut)", "Category Frequencies (Bar)"))
        
        fig.add_trace(go.Pie(labels=value_counts.index.astype(str), values=value_counts.values, hole=0.4,
                             marker=dict(colors=px.colors.qualitative.Safe), name="Donut"), row=1, col=1)
        
        fig.add_trace(go.Bar(x=value_counts.index.astype(str), y=value_counts.values, 
                             marker_color='#42A5F5', showlegend=False, name="Bar"), row=1, col=2)
        
        fig.update_layout(
            height=450,
            title_text=f"<b>Composition Analysis: Top 10 categories for {target_column}</b>",
            template="plotly_white",
            font=dict(family="Inter, sans-serif")
        )
        fig.show()


def plot_matrix(df: pd.DataFrame, columns=None, max_cols=5):
    """
    Plots a grid of all pairwise combinations of the selected columns (Generalised PairPlot).
    Contains univariate distributions on the diagonal, and bivariate relations off-diagonal.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Please provide a valid pandas DataFrame.")
        return

    # 1. Automatic Column Selection if not provided
    if columns is None:
        selected_cols = []
        for col in df.columns:
            # Skip high cardinality identifiers like names/ids/tickets
            col_data = df[col]
            cardinality = col_data.dropna().nunique()
            
            # Skip columns that have all NaNs
            if cardinality == 0:
                continue
                
            # Skip columns that are unique strings per row (e.g. PassengerId, Name)
            if col_data.dtype == 'object' or str(col_data.dtype).startswith('str'):
                if cardinality > 50 and cardinality / len(df) > 0.8:
                    continue
            
            selected_cols.append(col)
            
        # Select top max_cols
        columns = selected_cols[:max_cols]
        print(f"Auto-selected columns for visualization matrix: {columns}")
        print("You can customize this by passing a list of columns, e.g., plot(df, columns=['col1', 'col2'])")

    K = len(columns)
    if K == 0:
        print("Error: No plottable columns found in the dataset.")
        return

    # Check for invalid column names
    invalid_cols = [c for c in columns if c not in df.columns]
    if invalid_cols:
        print(f"Error: Columns {invalid_cols} do not exist in the DataFrame.")
        return

    # 2. Build Subplot Grid
    fig = make_subplots(
        rows=K, cols=K,
        shared_xaxes=False,
        shared_yaxes=False,
        vertical_spacing=0.08 / (K/5),
        horizontal_spacing=0.08 / (K/5)
    )

    # Classify all selected columns
    col_types = {c: _classify_column(df[c], c) for c in columns}

    # Sample data to make plotting efficient
    plot_df = df[columns].copy()
    plot_df = plot_df.sample(n=min(2000, len(plot_df)), random_state=42) if len(plot_df) > 2000 else plot_df

    # Set up trace colors
    c_diag_num = '#26A69A'  # Teal
    c_diag_cat = '#42A5F5'  # Blue
    c_diag_dt = '#AB47BC'   # Purple
    c_biv_num = '#00ACC1'   # Dark Teal
    c_biv_box = '#FF7043'   # Coral

    # 3. Add Traces to Subplots
    for r in range(1, K + 1):
        y_col = columns[r - 1]
        ty = col_types[y_col]
        
        for c in range(1, K + 1):
            x_col = columns[c - 1]
            tx = col_types[x_col]
            
            # DIAGONAL CELL
            if r == c:
                cell_series = plot_df[x_col].dropna()
                if len(cell_series) == 0:
                    continue
                if tx == 'numeric':
                    fig.add_trace(go.Histogram(x=cell_series, marker_color=c_diag_num, name=x_col, showlegend=False), row=r, col=c)
                elif tx == 'datetime':
                    ts = pd.to_datetime(cell_series, errors='coerce')
                    counts = ts.dt.to_period('M').value_counts().sort_index()
                    fig.add_trace(go.Scatter(x=counts.index.astype(str), y=counts.values, mode='lines+markers', line_color=c_diag_dt, name=x_col, showlegend=False), row=r, col=c)
                else: # categorical
                    counts = cell_series.value_counts().head(10)
                    fig.add_trace(go.Bar(x=counts.index.astype(str), y=counts.values, marker_color=c_diag_cat, name=x_col, showlegend=False), row=r, col=c)
            # OFF-DIAGONAL CELL
            else:
                # Isolate cell dataframe for off-diagonal (two distinct columns)
                cell_df = plot_df[[x_col, y_col]].dropna()
                if len(cell_df) == 0:
                    continue
                # Bivariate Numeric-Numeric -> Scatter
                if tx == 'numeric' and ty == 'numeric':
                    fig.add_trace(go.Scatter(
                        x=cell_df[x_col], y=cell_df[y_col], 
                        mode='markers', 
                        marker=dict(color=c_biv_num, opacity=0.6, size=5, line=dict(width=0.3, color='white')),
                        name=f"{x_col} vs {y_col}", showlegend=False
                    ), row=r, col=c)
                
                # Bivariate Numeric-Categorical -> Horizontal Box Plot
                elif tx == 'numeric' and ty == 'categorical':
                    fig.add_trace(go.Box(
                        x=cell_df[x_col], y=cell_df[y_col].astype(str), 
                        orientation='h', marker_color=c_biv_box, 
                        name=f"{x_col} vs {y_col}", showlegend=False
                    ), row=r, col=c)

                # Bivariate Categorical-Numeric -> Vertical Box Plot
                elif tx == 'categorical' and ty == 'numeric':
                    fig.add_trace(go.Box(
                        x=cell_df[x_col].astype(str), y=cell_df[y_col], 
                        orientation='v', marker_color=c_biv_box, 
                        name=f"{x_col} vs {y_col}", showlegend=False
                    ), row=r, col=c)

                # Bivariate Categorical-Categorical -> Heatmap
                elif tx == 'categorical' and ty == 'categorical':
                    ct = pd.crosstab(cell_df[y_col], cell_df[x_col])
                    fig.add_trace(go.Heatmap(
                        z=ct.values, x=ct.columns.astype(str), y=ct.index.astype(str), 
                        colorscale='Blues', showscale=False, name=f"{x_col} vs {y_col}"
                    ), row=r, col=c)

                # Bivariate Datetime-Numeric -> Trend Line
                elif tx == 'datetime' and ty == 'numeric':
                    ts = pd.to_datetime(cell_df[x_col], errors='coerce')
                    temp = pd.DataFrame({'date': ts.values, 'val': cell_df[y_col].values}).dropna()
                    temp['period'] = temp['date'].dt.to_period('M').astype(str)
                    trend = temp.groupby('period')['val'].mean().reset_index().sort_values('period')
                    fig.add_trace(go.Scatter(x=trend['period'], y=trend['val'], mode='lines+markers', line_color=c_diag_dt, name=f"{x_col} vs {y_col}", showlegend=False), row=r, col=c)
                
                # Bivariate Numeric-Datetime -> Trend Line (transposed coords)
                elif tx == 'numeric' and ty == 'datetime':
                    ts = pd.to_datetime(cell_df[y_col], errors='coerce')
                    temp = pd.DataFrame({'date': ts.values, 'val': cell_df[x_col].values}).dropna()
                    temp['period'] = temp['date'].dt.to_period('M').astype(str)
                    trend = temp.groupby('period')['val'].mean().reset_index().sort_values('period')
                    fig.add_trace(go.Scatter(x=trend['val'], y=trend['period'], mode='lines+markers', line_color=c_diag_dt, name=f"{x_col} vs {y_col}", showlegend=False), row=r, col=c)

                # Fallback -> Simple Count Heatmap
                else:
                    ct = pd.crosstab(cell_df[y_col].astype(str), cell_df[x_col].astype(str))
                    fig.add_trace(go.Heatmap(
                        z=ct.values, x=ct.columns.astype(str), y=ct.index.astype(str), 
                        colorscale='Blues', showscale=False, name=f"{x_col} vs {y_col}"
                    ), row=r, col=c)

            # Update cell labels
            if r == K:
                fig.update_xaxes(title_text=f"<b>{x_col}</b>", row=r, col=c)
            if c == 1:
                fig.update_yaxes(title_text=f"<b>{y_col}</b>", row=r, col=c)

    # 4. Global Layout Customizations
    grid_size = 180 * K + 150
    fig.update_layout(
        height=grid_size,
        width=grid_size,
        title_text="<b>ezclean Dataset Plot Matrix (All Combinations)</b>",
        template="plotly_white",
        font=dict(family="Inter, sans-serif", size=10),
        margin=dict(t=80, b=50, l=50, r=50)
    )
    
    fig.show()
    return fig


def plot_dashboard(df: pd.DataFrame, filename="ezclean_dashboard.html", show=True):
    """
    Creates an extremely interactive, fully self-contained HTML visual dashboard.
    Features descriptive statistics cards, a correlation heatmap, the pairplot matrix,
    and a dynamic javascript chart visualizer to explore column relationships offline.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Please provide a valid pandas DataFrame.")
        return

    # Calculate statistics for columns
    col_stats = []
    for col in df.columns:
        col_data = df[col]
        col_type = _classify_column(col_data, col)
        null_count = int(col_data.isna().sum())
        null_pct = round((null_count / len(df)) * 100, 2)
        unique_cnt = int(col_data.dropna().nunique())
        
        # Sample non-null values
        sample_vals = col_data.dropna().head(5).astype(str).tolist()
        sample_str = ", ".join(sample_vals)[:40]
        if len(col_data.dropna()) > 5:
            sample_str += "..."

        col_stats.append({
            "name": col,
            "type": col_type,
            "null_count": null_count,
            "null_pct": null_pct,
            "unique_count": unique_cnt,
            "sample": sample_str
        })

    # Sample dataset for the client-side JavaScript visualizer (up to 5000 rows for performance)
    sample_size = min(5000, len(df))
    sample_df = df.sample(n=sample_size, random_state=42) if len(df) > 5000 else df
    
    # We serialize the dataframe. For datetime, serialize to string ISO.
    # Convert datetime columns to string ISO first to avoid serialisation issues
    serializable_df = sample_df.copy()
    for col in serializable_df.columns:
        if pd.api.types.is_datetime64_any_dtype(serializable_df[col]):
            serializable_df[col] = serializable_df[col].astype(str)
        elif serializable_df[col].dtype == 'object' or str(serializable_df[col].dtype).startswith('str'):
            serializable_df[col] = serializable_df[col].astype(str).fillna("Unknown")
            
    dataset_json = serializable_df.to_json(orient='records')

    # Get pairplot matrix plotly JSON to inject
    try:
        # Pre-render the matrix layout for 4 columns max in the dashboard to prevent huge size
        matrix_fig = plot_matrix(df, max_cols=4)
        matrix_json = pio.to_json(matrix_fig)
    except Exception as e:
        print(f"Warning: Could not pre-generate pairplot matrix for dashboard: {e}")
        matrix_json = "{}"

    # General statistics
    num_rows = len(df)
    num_cols = len(df.columns)
    dup_rows = int(df.duplicated().sum())
    total_nulls = int(df.isna().sum().sum())
    null_percentage = round((total_nulls / (num_rows * num_cols)) * 100, 2) if (num_rows * num_cols) > 0 else 0

    # Read dashboard HTML template and inject JSON data
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ezclean - Dataset Explorer Dashboard</title>
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Plotly.js CDN -->
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {{ font-family: 'Inter', sans-serif; background-color: #f1f5f9; color: #1e293b; }}
        .tab-btn.active {{ border-bottom: 2px solid #2563eb; color: #2563eb; font-weight: 600; }}
        .glass-card {{ background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(8px); border: 1px solid rgba(226, 232, 240, 0.8); box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05); }}
    </style>
</head>
<body class="p-6">
    <div class="max-w-7xl mx-auto space-y-6">
        <!-- Header -->
        <header class="flex justify-between items-center p-6 glass-card rounded-2xl">
            <div>
                <h1 class="text-3xl font-bold tracking-tight text-slate-900">✨ ezclean Dashboard</h1>
                <p class="text-sm text-slate-500 mt-1">Interactive automated exploratory data dashboard</p>
            </div>
            <div class="bg-blue-50 text-blue-700 px-4 py-2 rounded-xl border border-blue-100 font-semibold text-sm">
                Active Dataset File
            </div>
        </header>

        <!-- KPI Grid -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div class="p-6 glass-card rounded-2xl flex flex-col">
                <span class="text-xs font-semibold text-slate-400 uppercase tracking-wider">Total Rows</span>
                <span class="text-3xl font-bold text-slate-800 mt-2">{num_rows:,}</span>
            </div>
            <div class="p-6 glass-card rounded-2xl flex flex-col">
                <span class="text-xs font-semibold text-slate-400 uppercase tracking-wider">Total Columns</span>
                <span class="text-3xl font-bold text-slate-800 mt-2">{num_cols}</span>
            </div>
            <div class="p-6 glass-card rounded-2xl flex flex-col">
                <span class="text-xs font-semibold text-slate-400 uppercase tracking-wider">Duplicate Rows</span>
                <span class="text-3xl font-bold text-slate-800 mt-2 { 'text-red-500' if dup_rows > 0 else 'text-emerald-500' }">{dup_rows}</span>
            </div>
            <div class="p-6 glass-card rounded-2xl flex flex-col">
                <span class="text-xs font-semibold text-slate-400 uppercase tracking-wider">Overall Missing Data</span>
                <span class="text-3xl font-bold text-slate-800 mt-2">{null_percentage}%</span>
                <span class="text-xs text-slate-400 mt-1">{total_nulls:,} missing cells</span>
            </div>
        </div>

        <!-- Navigation Tabs -->
        <div class="border-b border-slate-200 bg-white rounded-t-2xl px-6 flex space-x-6">
            <button onclick="switchTab('summary')" class="tab-btn py-4 text-sm font-medium text-slate-500 hover:text-slate-700 active" id="tab-summary">Dataset Summary</button>
            <button onclick="switchTab('matrix')" class="tab-btn py-4 text-sm font-medium text-slate-500 hover:text-slate-700" id="tab-matrix">Plot Matrix Grid</button>
            <button onclick="switchTab('explorer')" class="tab-btn py-4 text-sm font-medium text-slate-500 hover:text-slate-700" id="tab-explorer">Interactive Plotter</button>
        </div>

        <!-- Tab Contents -->
        <!-- Tab 1: Dataset Summary -->
        <div id="content-summary" class="tab-content block">
            <div class="glass-card rounded-b-2xl p-6 bg-white overflow-x-auto">
                <h3 class="text-lg font-semibold text-slate-800 mb-4">Column Definitions & Completeness</h3>
                <table class="w-full text-left border-collapse">
                    <thead>
                        <tr class="border-b border-slate-100 text-slate-400 text-xs font-semibold uppercase bg-slate-50/50">
                            <th class="py-3 px-4">Column Name</th>
                            <th class="py-3 px-4">Inferred Type</th>
                            <th class="py-3 px-4">Null Count</th>
                            <th class="py-3 px-4">Null Percentage</th>
                            <th class="py-3 px-4">Unique Cardinality</th>
                            <th class="py-3 px-4">Data Snippet</th>
                        </tr>
                    </thead>
                    <tbody class="divide-y divide-slate-100 text-sm">
                        <!-- Stats injected via string format -->
                        {"".join(f'''<tr class="hover:bg-slate-50/50 transition">
                            <td class="py-3 px-4 font-mono font-bold text-blue-600">{stat['name']}</td>
                            <td class="py-3 px-4"><span class="px-2.5 py-1 rounded-full text-xs font-semibold uppercase bg-slate-100 text-slate-600"> {stat['type']} </span></td>
                            <td class="py-3 px-4 text-slate-600">{stat['null_count']}</td>
                            <td class="py-3 px-4">
                                <div class="flex items-center space-x-2">
                                    <div class="w-16 bg-slate-100 rounded-full h-2">
                                        <div class="bg-red-400 h-2 rounded-full" style="width: {stat['null_pct']}%"></div>
                                    </div>
                                    <span class="text-xs text-slate-500 font-semibold">{stat['null_pct']}%</span>
                                </div>
                            </td>
                            <td class="py-3 px-4 text-slate-600 font-semibold">{stat['unique_count']}</td>
                            <td class="py-3 px-4 text-slate-400 italic font-mono text-xs">{stat['sample']}</td>
                        </tr>''' for stat in col_stats)}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Tab 2: Pairplot Matrix -->
        <div id="content-matrix" class="tab-content hidden">
            <div class="glass-card rounded-b-2xl p-6 bg-white flex justify-center">
                <div id="matrix-plot-container" class="w-full flex justify-center">
                    <div id="matrix-plot-div"></div>
                </div>
            </div>
        </div>

        <!-- Tab 3: Interactive Plotter -->
        <div id="content-explorer" class="tab-content hidden">
            <div class="grid grid-cols-1 md:grid-cols-4 gap-6">
                <!-- Control Panel -->
                <div class="glass-card rounded-2xl p-6 bg-white md:col-span-1 space-y-4">
                    <h3 class="text-base font-bold text-slate-800 uppercase tracking-wide">Plot Designer</h3>
                    
                    <div class="space-y-1.5">
                        <label class="text-xs font-semibold text-slate-400">X-Axis Column</label>
                        <select id="select-x" onchange="updateInteractivePlot()" class="w-full p-2.5 border border-slate-200 rounded-xl bg-slate-50 text-slate-700 font-medium">
                            {"".join(f'<option value="{stat["name"]}">{stat["name"]} ({stat["type"]})</option>' for stat in col_stats)}
                        </select>
                    </div>

                    <div class="space-y-1.5">
                        <label class="text-xs font-semibold text-slate-400">Y-Axis Column (Optional)</label>
                        <select id="select-y" onchange="updateInteractivePlot()" class="w-full p-2.5 border border-slate-200 rounded-xl bg-slate-50 text-slate-700 font-medium">
                            <option value="">-- None (Univariate) --</option>
                            {"".join(f'<option value="{stat["name"]}">{stat["name"]} ({stat["type"]})</option>' for stat in col_stats)}
                        </select>
                    </div>

                    <div class="pt-4 border-t border-slate-100">
                        <p class="text-xs text-slate-400 leading-relaxed">
                            💡 <b>Smart Plot Selection:</b> The plotter identifies data types and renders the best visualization matching your selection (Scatter, Box plot, Bar chart, or Timeline).
                        </p>
                    </div>
                </div>

                <!-- Plot Window -->
                <div class="glass-card rounded-2xl p-6 bg-white md:col-span-3 min-h-[450px]">
                    <div id="interactive-plot-div" class="w-full h-full min-h-[450px]"></div>
                </div>
            </div>
        </div>
    </div>

    <!-- Client-Side Dashboard Scripting -->
    <script>
        // Inject variables from python process
        const dataset = {dataset_json};
        const colTypes = {{
            {" ,".join(f'"{stat["name"]}": "{stat["type"]}"' for stat in col_stats)}
        }};
        const matrixPlotData = {matrix_json};

        // Tab Switching Logic
        function switchTab(tabId) {{
            // Hide all contents
            document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('block'));
            
            // Deactivate all buttons
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));

            // Show active content
            const activeContent = document.getElementById('content-' + tabId);
            activeContent.classList.remove('hidden');
            activeContent.classList.add('block');

            // Activate button
            document.getElementById('tab-' + tabId).classList.add('active');

            // Render/resize Plotly graphs if needed
            if (tabId === 'matrix' && matrixPlotData.data) {{
                Plotly.newPlot('matrix-plot-div', matrixPlotData.data, matrixPlotData.layout);
            }}
            
            // Recalculate plotter sizing
            if (tabId === 'explorer') {{
                updateInteractivePlot();
            }}
        }}

        // Client-side Chart Builder
        function updateInteractivePlot() {{
            const xCol = document.getElementById('select-x').value;
            const yCol = document.getElementById('select-y').value;
            
            const tx = colTypes[xCol];
            const ty = yCol ? colTypes[yCol] : null;

            // Extract arrays filtering out nulls
            const cleanData = dataset.filter(row => row[xCol] !== null && (!yCol || row[yCol] !== null));
            if (cleanData.length === 0) {{
                document.getElementById('interactive-plot-div').innerHTML = 
                    `<div class="flex items-center justify-center h-full text-slate-400 italic">No non-null data pairs available to plot.</div>`;
                return;
            }}

            const xVals = cleanData.map(row => row[xCol]);
            const yVals = yCol ? cleanData.map(row => row[yCol]) : null;

            let traces = [];
            let layout = {{
                template: 'plotly_white',
                font: {{ family: 'Inter, sans-serif' }},
                margin: {{ t: 40, b: 50, l: 60, r: 40 }}
            }};

            // CASE 1: Univariate (Y is not selected)
            if (!yCol) {{
                if (tx === 'numeric') {{
                    traces.push({{
                        x: xVals,
                        type: 'histogram',
                        name: xCol,
                        marker: {{ color: '#26A69A' }}
                    }});
                    layout.title = `<b>Histogram of ${{xCol}}</b>`;
                    layout.xaxis = {{ title: xCol }};
                    layout.yaxis = {{ title: 'Frequency' }};
                }} else if (tx === 'datetime') {{
                    // Count by Month
                    const counts = {{}};
                    xVals.forEach(val => {{
                        const dateStr = val.substring(0, 7); // YYYY-MM
                        counts[dateStr] = (counts[dateStr] || 0) + 1;
                    }});
                    const dates = Object.keys(counts).sort();
                    const freqs = dates.map(d => counts[d]);
                    
                    traces.push({{
                        x: dates,
                        y: freqs,
                        type: 'scatter',
                        mode: 'lines+markers',
                        line: {{ color: '#AB47BC', width: 3 }},
                        marker: {{ size: 7 }}
                    }});
                    layout.title = `<b>Timeline Count: Record volume of ${{xCol}} over time</b>`;
                    layout.xaxis = {{ title: 'Timeline' }};
                    layout.yaxis = {{ title: 'Volume Count' }};
                }} else {{
                    // Categorical Counts
                    const counts = {{}};
                    xVals.forEach(val => {{
                        counts[val] = (counts[val] || 0) + 1;
                    }});
                    const sortedLabels = Object.keys(counts).sort((a,b) => counts[b] - counts[a]).slice(0, 15);
                    const sortedValues = sortedLabels.map(l => counts[l]);

                    traces.push({{
                        x: sortedLabels,
                        y: sortedValues,
                        type: 'bar',
                        marker: {{ color: '#42A5F5' }}
                    }});
                    layout.title = `<b>Category Composition of ${{xCol}} (Top 15)</b>`;
                    layout.xaxis = {{ title: xCol }};
                    layout.yaxis = {{ title: 'Count' }};
                }}
            }}
            // CASE 2: Bivariate (Y is selected)
            else {{
                if (tx === 'numeric' && ty === 'numeric') {{
                    // Scatter Plot
                    traces.push({{
                        x: xVals,
                        y: yVals,
                        mode: 'markers',
                        type: 'scatter',
                        marker: {{ color: '#00ACC1', opacity: 0.7, size: 7 }}
                    }});
                    layout.title = `<b>Scatter Plot: ${{yCol}} vs ${{xCol}}</b>`;
                    layout.xaxis = {{ title: xCol }};
                    layout.yaxis = {{ title: yCol }};
                }} else if (tx === 'numeric' && ty === 'categorical') {{
                    // Horizontal Box Plot
                    traces.push({{
                        x: xVals,
                        y: yVals.map(String),
                        type: 'box',
                        orientation: 'h',
                        marker: {{ color: '#FF7043' }}
                    }});
                    layout.title = `<b>Horizontal Box Distribution: ${{xCol}} grouped by ${{yCol}}</b>`;
                    layout.xaxis = {{ title: xCol }};
                    layout.yaxis = {{ title: yCol }};
                }} else if (tx === 'categorical' && ty === 'numeric') {{
                    // Vertical Box Plot
                    traces.push({{
                        x: xVals.map(String),
                        y: yVals,
                        type: 'box',
                        orientation: 'v',
                        marker: {{ color: '#FF7043' }}
                    }});
                    layout.title = `<b>Vertical Box Distribution: ${{yCol}} grouped by ${{xCol}}</b>`;
                    layout.xaxis = {{ title: xCol }};
                    layout.yaxis = {{ title: yCol }};
                }} else if (tx === 'categorical' && ty === 'categorical') {{
                    // Stacked Bar or Heatmap. Let's do Heatmap (like Crosstab)
                    const labelX = [...new Set(xVals)].slice(0, 15);
                    const labelY = [...new Set(yVals)].slice(0, 15);
                    
                    const matrix = Array(labelY.length).fill().map(() => Array(labelX.length).fill(0));
                    
                    cleanData.forEach(row => {{
                        const idxX = labelX.indexOf(row[xCol]);
                        const idxY = labelY.indexOf(row[yCol]);
                        if (idxX !== -1 && idxY !== -1) {{
                            matrix[idxY][idxX]++;
                        }}
                    }});

                    traces.push({{
                        z: matrix,
                        x: labelX.map(String),
                        y: labelY.map(String),
                        type: 'heatmap',
                        colorscale: 'Blues',
                        showscale: true
                    }});
                    layout.title = `<b>Contingency Table Heatmap: ${{yCol}} vs ${{xCol}} (Top 15)</b>`;
                    layout.xaxis = {{ title: xCol }};
                    layout.yaxis = {{ title: yCol }};
                }} else if (tx === 'datetime' && ty === 'numeric') {{
                    // Trend Plot
                    const sum = {{}};
                    const count = {{}};
                    cleanData.forEach(row => {{
                        const dt = row[xCol].substring(0, 7); // month
                        sum[dt] = (sum[dt] || 0) + Number(row[yCol]);
                        count[dt] = (count[dt] || 0) + 1;
                    }});
                    const dates = Object.keys(sum).sort();
                    const averages = dates.map(d => sum[d] / count[d]);

                    traces.push({{
                        x: dates,
                        y: averages,
                        type: 'scatter',
                        mode: 'lines+markers',
                        line: {{ color: '#AB47BC', width: 3 }},
                        marker: {{ size: 7 }}
                    }});
                    layout.title = `<b>Timeline Trend: Average ${{yCol}} over time</b>`;
                    layout.xaxis = {{ title: xCol }};
                    layout.yaxis = {{ title: `Mean ${{yCol}}` }};
                }} else {{
                    // Fallback Simple Scatter
                    traces.push({{
                        x: xVals.map(String),
                        y: yVals.map(String),
                        mode: 'markers',
                        type: 'scatter',
                        marker: {{ color: '#90a4ae', size: 8 }}
                    }});
                    layout.title = `<b>Comparison Matrix: ${{yCol}} vs ${{xCol}}</b>`;
                    layout.xaxis = {{ title: xCol }};
                    layout.yaxis = {{ title: yCol }};
                }}
            }}

            Plotly.newPlot('interactive-plot-div', traces, layout);
        }}

        // Render default summary tab on load
        window.onload = function() {{
            switchTab('summary');
        }};
    </script>
</body>
</html>
"""

    # Save to file
    out_path = os.path.abspath(filename)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Interactive dashboard successfully generated: {out_path}")
    if show:
        webbrowser.open(f"file:///{out_path.replace(os.sep, '/')}")
        
    return out_path


def plot(df: pd.DataFrame, target_column: str = None, columns=None):
    """
    Unified entrypoint plot caller. 
    If a target_column string is provided, runs dedicated column distribution.
    If target_column is None, plots a generalised PairPlot matrix of subplots for all columns.
    """
    if target_column is not None:
        plot_single_column(df, target_column)
    else:
        plot_matrix(df, columns=columns)