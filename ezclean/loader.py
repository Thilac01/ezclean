import pandas as pd
import numpy as np
import json
import csv
import requests as req
import time as t
from io import StringIO

class Smart_loader:
    
    def __new__(cls, file_path):
        """
        Special constructor method that directly returns a raw 
        Pandas DataFrame object the moment the class is called.
        """
        # 1. Your custom backward-reading character loop to isolate the extension
        extension_letters = []
        for char in reversed(file_path):
            if char == '.':
                break 
            extension_letters.append(char)
            
        extension_letters.reverse()
        extention = "".join(extension_letters).lower()
        
        # Sector master list of supported formats
        EXTENTION = [
            "csv", "tsv", "txt", "fwf", 
            "json", "jsonl", "ndjson", "xml", "html",
            "xlsx", "xls", "xlsm", "xlsb", "ods",
            "parquet", "pq", "feather", "arrow", "orc", "avro", "hdf5", "h5"
        ]

        # 2. Match extensions and route straight to raw DataFrames
        if extention in EXTENTION:
            
            # Web Request Routing
            if file_path.startswith("http") or extention == 'html':
                response = req.get(file_path)
                if response.status_code != 200:
                    raise ConnectionError(f"Could not download data from URL. HTTP Code: {response.status_code}")
                
                if extention == 'csv':
                    return pd.read_csv(StringIO(response.text))
                elif extention == 'json':
                    return pd.read_json(StringIO(response.text))
                elif extention == 'html':
                    return pd.read_html(StringIO(response.text))[0]
            
            # Local Storage File Routing
            else:
                if extention == 'csv':
                    return pd.read_csv(file_path)
                elif extention == 'json':
                    return pd.read_json(file_path)
                elif extention in ['xlsx', 'xls']:
                    return pd.read_excel(file_path)
                elif extention == 'parquet':
                    return pd.read_parquet(file_path)
                else:
                    raise NotImplementedError(f"Extension '.{extention}' recognized but reader logic is not assigned yet.")
        else:
            raise ValueError(f"This file type is not yet implemented. Please use: {EXTENTION[:5]}")