import os
import pandas as pd
import numpy as np
import requests as req
from io import StringIO, BytesIO
from urllib.parse import urlparse

class Smart_loader:
    
    def __new__(cls, file_path, **kwargs):
        """
        Universal data loader interceptor. Map extensions directly 
        to their respective optimized Pandas reader engines.
        """
        # 1. Isolate extension safely (ignores URL parameters like ?token=xyz)
        pure_path = urlparse(file_path).path
        ext = os.path.splitext(pure_path)[1].replace('.', '').lower()
        
        # Handle exceptions where standard extensions don't match exactly
        if ext in ['txt', 'tsv']:
            ext = 'csv'
        elif ext in ['xlsx', 'xls', 'xlsm', 'xlsb', 'ods', 'odt']:
            ext = 'excel'
        elif ext in ['pq']:
            ext = 'parquet'
        elif ext in ['h5']:
            ext = 'hdf'
        elif ext in ['pkl']:
            ext = 'pickle'

        # 2. Define the central engine router mapping
        # Maps extensions to (Local Reader Function, Buffer Reader Function)
        ENGINE_MAP = {
            "csv":     (lambda p, **kw: pd.read_csv(p, **kw),       lambda b, **kw: pd.read_csv(StringIO(b.decode('utf-8', errors='ignore')), **kw)),
            "json":    (lambda p, **kw: pd.read_json(p, **kw),      lambda b, **kw: pd.read_json(StringIO(b.decode('utf-8', errors='ignore')), **kw)),
            "jsonl":   (lambda p, **kw: pd.read_json(p, lines=True, **kw), lambda b, **kw: pd.read_json(StringIO(b.decode('utf-8', errors='ignore')), lines=True, **kw)),
            "ndjson":  (lambda p, **kw: pd.read_json(p, lines=True, **kw), lambda b, **kw: pd.read_json(StringIO(b.decode('utf-8', errors='ignore')), lines=True, **kw)),
            "excel":   (lambda p, **kw: pd.read_excel(p, **kw),     lambda b, **kw: pd.read_excel(BytesIO(b), **kw)),
            "parquet": (lambda p, **kw: pd.read_parquet(p, **kw),   lambda b, **kw: pd.read_parquet(BytesIO(b), **kw)),
            "feather": (lambda p, **kw: pd.read_feather(p, **kw),   lambda b, **kw: pd.read_feather(BytesIO(b), **kw)),
            "arrow":   (lambda p, **kw: pd.read_feather(p, **kw),   lambda b, **kw: pd.read_feather(BytesIO(b), **kw)),
            "orc":     (lambda p, **kw: pd.read_orc(p, **kw),       lambda b, **kw: pd.read_orc(BytesIO(b), **kw)),
            "xml":     (lambda p, **kw: pd.read_xml(p, **kw),       lambda b, **kw: pd.read_xml(BytesIO(b), **kw)),
            "html":    (lambda p, **kw: pd.read_html(p, **kw)[0],   lambda b, **kw: pd.read_html(StringIO(b.decode('utf-8', errors='ignore')), **kw)[0]),
            "fwf":     (lambda p, **kw: pd.read_fwf(p, **kw),       lambda b, **kw: pd.read_fwf(StringIO(b.decode('utf-8', errors='ignore')), **kw)),
            "pickle":  (lambda p, **kw: pd.read_pickle(p, **kw),    lambda b, **kw: pd.read_pickle(BytesIO(b), **kw)),
            "stata":   (lambda p, **kw: pd.read_stata(p, **kw),     lambda b, **kw: pd.read_stata(BytesIO(b), **kw)),
            "spss":    (lambda p, **kw: pd.read_spss(p, **kw),      lambda b, **kw: pd.read_spss(BytesIO(b), **kw)),
            "sas":     (lambda p, **kw: pd.read_sas(p, **kw),       lambda b, **kw: pd.read_sas(BytesIO(b), **kw)),
            "hdf":     (lambda p, **kw: pd.read_hdf(p, **kw),       None), # HDF5 requires local file access / disk handles
        }

        if ext not in ENGINE_MAP:
            raise ValueError(f"Extension '.{ext}' is unsupported. Choose from: {list(ENGINE_MAP.keys())}")

        local_reader, web_reader = ENGINE_MAP[ext]

        # 3. Execution Path: Web Fetching vs Local File Access
        if file_path.startswith(("http://", "https://")):
            if web_reader is None:
                raise NotImplementedError(f"Format '.{ext}' cannot be streamed directly over raw HTTP. Download locally first.")
                
            response = req.get(file_path, timeout=15)
            if response.status_code != 200:
                raise ConnectionError(f"HTTP Target unreachable. Status: {response.status_code}")
                
            return web_reader(response.content, **kwargs)
            
        else:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Local file target missing: '{file_path}'")
            return local_reader(file_path, **kwargs)