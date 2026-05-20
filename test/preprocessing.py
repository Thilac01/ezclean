import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OrdinalEncoder, LabelEncoder

class DataPreprocessor:
    @staticmethod
    def scale_numeric(df, method='standard', columns=None):
        """
        Numeric: Min-Max, Standard (Z-score), and Robust scaling.
        """
        df_scaled = df.copy()
        if columns is None:
            columns = df_scaled.select_dtypes(include=[np.number]).columns

        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("Unsupported scaling method. Choose 'standard', 'minmax', or 'robust'.")

        if len(columns) > 0:
            df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
            
        return df_scaled

    @staticmethod
    def encode_categorical(df, method='onehot', columns=None):
        """
        Categorical: One-Hot, Ordinal, and Uniform (Label) encoding.
        """
        df_encoded = df.copy()
        if columns is None:
            columns = df_encoded.select_dtypes(include=['object', 'category']).columns

        if method == 'onehot':
            df_encoded = pd.get_dummies(df_encoded, columns=columns, drop_first=False)
        elif method == 'ordinal':
            encoder = OrdinalEncoder()
            if len(columns) > 0:
                df_encoded[columns] = encoder.fit_transform(df_encoded[columns].astype(str))
        elif method == 'uniform' or method == 'label':
            for col in columns:
                encoder = LabelEncoder()
                df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
        else:
            raise ValueError("Unsupported encoding method. Choose 'onehot', 'ordinal', or 'uniform'.")
            
        return df_encoded
