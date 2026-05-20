import pandas as pd
import numpy as np

class DataInspector:
    @staticmethod
    def inspect(df):
        """
        Analyzes a DataFrame and prints a structured health and profile report.
        Returns a summary DataFrame for programmatic use.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("DataInspector requires a valid Pandas DataFrame as input.")

        if df.empty:
            print("⚠️ The provided DataFrame is completely empty.")
            return pd.DataFrame()

        total_rows, total_cols = df.shape
        total_cells = total_rows * total_cols
        total_nulls = df.isnull().sum().sum()
        null_percentage = (total_nulls / total_cells) * 100 if total_cells > 0 else 0
        memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

        print("=" * 60)
        print("          *** EZCLEAN AUTOMATED DATA PROFILE ***          ")
        print("=" * 60)
        print(f"[I] Dataset Dimensions : {total_rows} Rows | {total_cols} Columns")
        print(f"[+] Total Data Points  : {total_cells}")
        print(f"[!] Missing/Null Cells : {total_nulls} ({null_percentage:.2f}%)")
        print(f"[$] Memory Allocation  : {memory_usage_mb:.2f} MB")
        print("-" * 60)
        print(f"{'COLUMN NAME':<22} | {'TYPE':<12} | {'NULLS':<7} | {'NULL %':<7} | {'UNIQUE':<6}")
        print("-" * 60)

        report_records = []
        
        for col in df.columns:
            dtype_str = str(df[col].dtype)
            null_count = df[col].isnull().sum()
            null_pct = (null_count / total_rows) * 100 if total_rows > 0 else 0
            unique_count = df[col].nunique()
            
            # Print row visually formatted to align perfectly
            print(f"{col[:22]:<22} | {dtype_str[:12]:<12} | {null_count:<7} | {null_pct:.1f}%{'' : <3} | {unique_count:<6}")
            
            report_records.append({
                "Column Name": col,
                "Data Type": dtype_str,
                "Missing Values": null_count,
                "Missing %": f"{null_pct:.2f}%",
                "Unique Values": unique_count
            })
            
        print("=" * 60)
        
        # Return summary as a standard DataFrame in case the developer wants to manipulate it
        return pd.DataFrame(report_records)