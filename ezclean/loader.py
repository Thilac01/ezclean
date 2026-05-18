import os
import io
import pandas as pd
import numpy as np
import requests
from pypdf import PdfReader

class SmartLoader:
    def __init__(self, default_na_values=None):
        """
        Initializes the loader with an aggressive, comprehensive list of real-world 
        and messy null string representations commonly found in dirty datasets.
        """
        if default_na_values is None:
            self.na_values = [
                '?', 'N/A', 'n/a', 'NA', 'na', 'NULL', 'null', 
                'empty', '-', '--', 'NaN', 'nan', 'none', 'None', 
                'inf', '-inf', 'missing', 'null_value', 'void'
            ]
        else:
            self.na_values = default_na_values

    def load_data(self, source, **kwargs):
        """
        The Ultimate Universal Entry Point. Converts any incoming data source 
        into a highly optimized Pandas DataFrame.
        
        Accepts:
        - Local System Paths (.csv, .xlsx, .xls, .json, .html, .pdf)
        - Web URLs (Direct HTTP/HTTPS links to any supported file format)
        - Raw Python collections (Dictionaries, lists of dicts from API responses)
        - Live Pandas DataFrames (Passes straight through for automated type optimization)
        """
        # Inject standard/custom missing values into user configuration arguments
        kwargs['na_values'] = kwargs.get('na_values', []) + self.na_values

        # CASE 1: Source is already an existing Pandas DataFrame
        if isinstance(source, pd.DataFrame):
            return self._auto_optimize_types(source.copy())

        # CASE 2: Source is raw API data (dict or list of objects)
        if isinstance(source, (dict, list)):
            df = pd.DataFrame(source)
            return self._auto_optimize_types(df)

        # CASE 3: Source is a reference string (File path or Web Endpoint URL)
        if isinstance(source, str):
            if source.startswith(('http://', 'https://')):
                return self._load_from_url(source, **kwargs)
            if os.path.exists(source):
                return self._load_from_local_file(source, **kwargs)
            
            raise FileNotFoundError(f"Provided source string is neither a valid URL nor a found local file: {source}")

        raise TypeError(f"Unsupported input type: {type(source)}. Input data as a path string, URL, dict, list, or DataFrame.")

    def _load_from_url(self, url, **kwargs):
        """Downloads external files or web tables straight into memory streams."""
        response = requests.get(url)
        response.raise_for_status()
        
        # Isolate base path extensions from tracking query variables
        clean_path = url.split('?')[0]
        ext = os.path.splitext(clean_path)[1].lower()

        if ext == '.csv':
            df = pd.read_csv(io.StringIO(response.text), **kwargs)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(io.BytesIO(response.content), **kwargs)
        elif ext == '.json':
            df = pd.read_json(io.StringIO(response.text), **kwargs)
        elif ext in ['.html', '.htm'] or response.headers.get('Content-Type', '').startswith('text/html'):
            # Pulls all tables from a web URL and merges them, or selects the largest one
            dfs = pd.read_html(io.StringIO(response.text), **kwargs)
            df = max(dfs, key=len) if dfs else pd.DataFrame()
        else:
            # Fallback: Try reading it as a standard CSV text stream
            try:
                df = pd.read_csv(io.StringIO(response.text), **kwargs)
            except Exception:
                raise ValueError(f"Unable to parse or auto-identify streaming web structure from URL: {url}")
                
        return self._auto_optimize_types(df)

    def _load_from_local_file(self, filepath, **kwargs):
        """Identifies file formats and extracts structural data into a DataFrame."""
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == '.csv':
            df = pd.read_csv(filepath, **kwargs)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath, **kwargs)
        elif ext == '.json':
            df = pd.read_json(filepath, **kwargs)
        elif ext in ['.html', '.htm']:
            dfs = pd.read_html(filepath, **kwargs)
            df = max(dfs, key=len) if dfs else pd.DataFrame()
        elif ext == '.pdf':
            df = self._load_pdf_as_dataframe(filepath)
        else:
            raise ValueError(f"Extension format '{ext}' is currently unsupported by ezclean.")
            
        return self._auto_optimize_types(df)

    def _load_pdf_as_dataframe(self, filepath):
        """Scrapes text layout matrices from PDF files and aligns them into tabular records."""
        reader = PdfReader(filepath)
        all_rows = []
        
        for page in reader.pages:
            text = page.extract_text()
            if not text:
                continue
            lines = text.strip().split('\n')
            for line in lines:
                if ',' in line:
                    row = [v.strip() for v in line.split(',')]
                elif ';' in line:
                    row = [v.strip() for v in line.split(';')]
                else:
                    row = [v.strip() for v in line.split('  ') if v.strip()]
                if row:
                    all_rows.append(row)
                    
        if not all_rows:
            return pd.DataFrame()
            
        header, data_rows = all_rows[0], all_rows[1:]
        max_cols = len(header)
        
        # Pad short rows with missing values dynamically to ensure seamless DF alignment
        clean_rows = [r + [np.nan]*(max_cols-len(r)) if len(r) < max_cols else r[:max_cols] for r in data_rows]
        return pd.DataFrame(clean_rows, columns=header)

    def _auto_optimize_types(self, df):
        """Normalizes strings, registers computational NaNs, downcasts numbers, optimizes categories."""
        if df.empty:
            return df
            
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace(self.na_values, np.nan)
            
            # Auto-downcast numeric columns to save massive system memory
            converted_numeric = pd.to_numeric(df[col], errors='ignore')
            if converted_numeric.dtype in ['int64', 'float64']:
                if converted_numeric.dtype == 'float64':
                    df[col] = pd.to_numeric(converted_numeric, downcast='float')
                else:
                    df[col] = pd.to_numeric(converted_numeric, downcast='integer')
                continue
            
            # Low Cardinality Category conversion
            if df[col].dtype == 'object':
                num_unique = df[col].nunique()
                total_rows = len(df)
                if total_rows > 0 and (num_unique / total_rows) < 0.12:
                    df[col] = df[col].astype('category')
                    
        return df