import pandas as pd
import numpy as np
import os
from ezclean import (
    SmartLoader, 
    DataInspector, 
    DataCleaner, 
    DataPreprocessor, 
    Visualizer, 
    StatInsights,
    AIPipeline,
    ReportGenerator
)

def main():
    # 1. Simulate a real-world messy dataset dictionary
    dirty_data = {
        'Transaction_ID': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120],
        'Customer_Age': ['34', ' ? ', '45', '22', 'N/A', '60', '34', '1000', '29', '42', '55', '31', '38', '47', '21', '62', '28', '33', '37', '41'], 
        'Product_Category': ['Electronics', 'Clothing', 'Electronics', 'Clothing', 'Electronics', 'Home', 'Clothing', 'Electronics', 'Home', 'Electronics', 'Clothing', 'Home', 'Electronics', 'Clothing', 'Home', 'Electronics', 'Clothing', 'Home', 'Electronics', 'Clothing'],
        'Revenue': [150.50, np.nan, 200.00, 45.00, 99.99, 300.00, 45.00, 99.99, 120.0, 250.0, 60.0, 310.0, 180.0, 55.0, 110.0, 220.0, 80.0, 290.0, 175.0, 65.0],
        'Rating': [5, 3, 4, np.nan, 2, 5, 3, 4, 4, 5, 3, 5, 4, 3, 4, 5, 3, 4, 4, 3],
        'Is_Fraud': [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  # Target variable
    }

    print("=== 1. LOADING DATA ===")
    loader = SmartLoader()
    df = loader.load_data(dirty_data)
    print("Data loaded successfully.")

    print("\n=== 2. DATA INSPECTION ===")
    DataInspector.inspect(df)

    print("\n=== 3. AUTOMATED CLEANING ===")
    df_clean = DataCleaner.impute_missing(df, strategy='mean', columns=['Revenue', 'Rating'])
    df_clean = DataCleaner.impute_missing(df_clean, strategy='median', columns=['Customer_Age'])
    df_clean = DataCleaner.remove_duplicates(df_clean)
    df_clean = DataCleaner.remove_outliers_iqr(df_clean, columns=['Customer_Age'])
    print("Cleaned DataFrame Sample:")
    print(df_clean.head())

    print("\n=== 4. AI FEATURE ENGINEERING (QUALITY INDEX) ===")
    # Calculate a composite mathematical index using sigmoid activation function
    df_clean = AIPipeline.calculate_quality_index(df_clean, columns=['Customer_Age', 'Revenue', 'Rating'], activation='sigmoid')
    print("DataFrame with AI EZ_Quality_Index formulated:")
    print(df_clean[['Transaction_ID', 'Revenue', 'EZ_Quality_Index']].head())

    print("\n=== 5. AUTO-ML TRAINING ===")
    # Train a fast ML model on the cleaned data
    model, score, metric_name, task = AIPipeline.auto_train(df_clean, target_col='Is_Fraud', task_type='classification')
    print(f"Auto-ML Task: {task}")
    print(f"Model: RandomForest - {metric_name}: {score:.4f}")

    print("\n=== 6. REPORT GENERATION ===")
    latex_report = ReportGenerator.generate_latex_report(
        df=df_clean, 
        dataset_name="Retail Transaction Dataset",
        target_col='Is_Fraud',
        ai_score=score,
        ai_metric=metric_name
    )
    
    with open("final_report.tex", "w") as f:
        f.write(latex_report)
        
    print(f"LaTeX report successfully written to: {os.path.abspath('final_report.tex')}")

if __name__ == "__main__":
    main()