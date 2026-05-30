import pandas as pd
import numpy as np
import re

class DataCleaner:
    """
    A versatile, dataset-agnostic data processing engine designed to clean,
    sanitize, impute, and handle outliers across any structured DataFrame.
    """
    def __init__(self, df: pd.DataFrame):
        """Initializes with a copy of a DataFrame to protect original source arrays."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Data engine target must be a valid pandas DataFrame instance.")
        self.df = df.copy()
        self.garbage_strings = ['?', 'n/a', 'NULL', ' ', 'NA', 'null', 'NaN', '-', 'None', 'nil']

    def column_name_sanity(self):
        """Standardizes column matrices to snake_case stripping symbols and whitespace."""
        def clean_name(name):
            s = str(name).strip().replace(" ", "_")
            s = re.sub(r'[^\w\s_]', '', s) # Strip punctuations
            return re.sub(r'_+', '_', s).lower() # Deduplicate underscores and lowercase
            
        self.df.columns = [clean_name(col) for col in self.df.columns]
        return self

    def sanitize_data(self):
        """Replaces diverse structural garbage tokens with true NumPy NaN structures."""
        self.df.replace(self.garbage_strings, np.nan, inplace=True)
        return self

    def text_normalization(self):
        """Cleans object/string vectors by stripping extreme whitespaces and padding fragments."""
        for col in self.df.select_dtypes(include=['object', 'category']).columns:
            # Convert values safely to string, strip whitespace, revert string NaNs to real NaNs
            self.df[col] = self.df[col].astype(str).str.strip()
            self.df[col] = self.df[col].replace(['nan', 'NaN', 'None', '<NA>'], np.nan)
        return self

    def auto_type_correction(self):
        """Forces column datatypes to float/integer schemas if conversions maintain integrity."""
        for col in self.df.columns:
            if self.df[col].dtype == 'object' or str(self.df[col].dtype).startswith('str'):
                # Try converting to numeric
                converted = pd.to_numeric(self.df[col], errors='coerce')
                
                # SMART THRESHOLD: If more than 50% of the non-null data successfully converts to numbers,
                # then this column is supposed to be numeric!
                original_non_nulls = self.df[col].dropna().shape[0]
                converted_non_nulls = converted.dropna().shape[0]
                
                if original_non_nulls > 0 and (converted_non_nulls / original_non_nulls) > 0.5:
                    self.df[col] = converted
                    
        # Fallback date-time parsing for typical calendar metrics
        for col in self.df.columns:
            if 'date' in str(col).lower() or 'time' in str(col).lower():
                try:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                except Exception:
                    pass
        return self

    def remove_duplicates(self):
        """Prunes exact row duplications from the dataframe workspace."""
        self.df.drop_duplicates(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        return self

    def intelligent_null_filling(self):
        """Imputes missing data values dynamically based on underlying array data types."""
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        cat_cols = self.df.select_dtypes(exclude=[np.number, 'datetime64[ns]']).columns
        
        # Numeric attributes fill using robust median metric
        for col in num_cols:
            if self.df[col].isna().sum() > 0:
                self.df[col] = self.df[col].fillna(self.df[col].median())
                
        # Categorical labels fill using structural standard string placeholders
        for col in cat_cols:
            if self.df[col].isna().sum() > 0:
                # Fallback to structural standard string placeholder
                self.df[col] = self.df[col].fillna("Unknown")
        return self

    def handle_outliers(self, outlier_config):
        """
        Detects and drops row matrices breaking the targeted IQR threshold boundary box scale.
        Accepts formats: True (defaults to 1.5 * IQR) or (True, custom_multiplier).
        """
        # Parse variable parameter tuple settings
        enabled = False
        multiplier = 1.5
        
        if isinstance(outlier_config, bool):
            enabled = outlier_config
        elif isinstance(outlier_config, tuple) and len(outlier_config) == 2:
            enabled, multiplier = outlier_config
            
        if not enabled:
            return self

        num_cols = self.df.select_dtypes(include=[np.number]).columns
        mask = pd.Series(False, index=self.df.index)
        
        for col in num_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            # Identify outlier positions
            col_outliers = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            mask = mask | col_outliers
            
        self.df = self.df[~mask].reset_index(drop=True)
        return self

    def get_dataframe(self) -> pd.DataFrame:
        """Exposes the internal modified dataframe structure array directly."""
        return self.df


# Functional API Interface Mapping Wrapper
def Cleaner(
    df: pd.DataFrame, 
    col_sanity: bool = True,
    sanitize: bool = True,
    text_norm: bool = True,
    type_correct: bool = True,
    duplicate: bool = True,
    null_fill: bool = True,
    outlayer = True  # Can take True, False, or a tuple such as (True, 5)
) -> pd.DataFrame:
    """
    Unified high-level pipeline wrapper. Passing parameters as False skips specific modules.
    """
    engine = DataCleaner(df)
    
    if col_sanity:   engine.column_name_sanity()
    if sanitize:     engine.sanitize_data()
    if text_norm:    engine.text_normalization()
    if type_correct: engine.auto_type_correction()
    if duplicate:    engine.remove_duplicates()
    if null_fill:    engine.intelligent_null_filling()
    if outlayer:     engine.handle_outliers(outlayer)
        
    return engine.get_dataframe()
