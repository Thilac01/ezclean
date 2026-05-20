import pandas as pd
import numpy as np
from .cleaner import DataCleaner
from .preprocessing import DataPreprocessor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score

class AIPipeline:
    @staticmethod
    def calculate_quality_index(df, columns=None, activation='sigmoid'):
        """
        Creates a new composite index using a mathematical activation function.
        Useful for creating normalized feature scores.
        """
        df_new = df.copy()
        if columns is None:
            columns = df_new.select_dtypes(include=[np.number]).columns
            
        if len(columns) == 0:
            return df_new

        # Compute a raw linear combination (sum of standardized features)
        raw_score = np.zeros(len(df_new))
        for col in columns:
            col_min, col_max = df_new[col].min(), df_new[col].max()
            if col_max - col_min != 0:
                # scale to -1 to 1 roughly
                scaled = 2 * ((df_new[col] - col_min) / (col_max - col_min)) - 1
                raw_score += scaled
                
        # Apply activation function to formulate the final index
        if activation == 'sigmoid':
            df_new['EZ_Quality_Index'] = 1 / (1 + np.exp(-raw_score))
        elif activation == 'tanh':
            df_new['EZ_Quality_Index'] = np.tanh(raw_score)
        elif activation == 'relu':
            df_new['EZ_Quality_Index'] = np.maximum(0, raw_score)
        else:
            df_new['EZ_Quality_Index'] = raw_score
            
        return df_new

    @staticmethod
    def auto_train(df, target_col, task_type='auto'):
        """
        Full pack AI/ML: Automatically determines task, splits data, and trains a Random Forest model.
        Returns the trained model and the main metric score (Accuracy or R2).
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
            
        df_clean = df.dropna(subset=[target_col]).copy()
        
        # Separate X and y
        X = df_clean.drop(columns=[target_col])
        y = df_clean[target_col]
        
        # Identify categorical columns and encode them automatically
        cat_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
            
        # Impute remaining missing values in X
        X = X.fillna(X.mean(numeric_only=True))
        X = X.fillna(0) # For any remaining NaNs
        
        # Determine task type
        if task_type == 'auto':
            if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
                task_type = 'regression'
            else:
                task_type = 'classification'
                
        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if task_type == 'regression':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = r2_score(y_test, preds)
            metric_name = "R2 Score"
        else:
            # Classification
            y_train = y_train.astype(str)
            y_test = y_test.astype(str)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = accuracy_score(y_test, preds)
            metric_name = "Accuracy"
            
        return model, score, metric_name, task_type
