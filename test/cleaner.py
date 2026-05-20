import pandas as pd
import numpy as np

class DataCleaner:
    @staticmethod
    def impute_missing(df, strategy='mean', fill_value=None, columns=None):
        """
        Identify and impute missing values using mean, median, mode, or constant strategies.
        """
        df_clean = df.copy()
        if columns is None:
            columns = df_clean.columns

        for col in columns:
            if df_clean[col].isnull().any():
                if strategy == 'constant':
                    if fill_value is None:
                        raise ValueError("fill_value must be provided for constant strategy")
                    df_clean[col] = df_clean[col].fillna(fill_value)
                else:
                    if pd.api.types.is_numeric_dtype(df_clean[col]):
                        if strategy == 'mean':
                            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                        elif strategy == 'median':
                            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                        elif strategy == 'mode':
                            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
                        else:
                            raise ValueError(f"Unsupported strategy '{strategy}' for numeric column '{col}'")
                    else:
                        if strategy == 'mode':
                            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
                        elif strategy == 'constant':
                            pass # Handled above
                        else:
                            raise ValueError(f"Strategy '{strategy}' not supported for categorical column '{col}'. Use 'mode' or 'constant'.")
        return df_clean

    @staticmethod
    def remove_duplicates(df, subset=None, keep='first'):
        """
        Remove exact duplicate rows.
        """
        return df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)

    @staticmethod
    def remove_outliers_iqr(df, columns=None, multiplier=1.5):
        """
        Remove specific outliers using IQR logic.
        """
        df_clean = df.copy()
        if columns is None:
            columns = df_clean.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
                
        return df_clean.reset_index(drop=True)

    @staticmethod
    def drop_rows(df, indices):
        """
        Interactive row deletion (by index).
        """
        return df.drop(index=indices).reset_index(drop=True)

    @staticmethod
    def drop_columns(df, columns):
        """
        Interactive column deletion.
        """
        return df.drop(columns=columns)
