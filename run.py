import os
from ezclean import *

def main():
    # 1. Load and Clean the dataset
    df = Smart_loader("tested.csv")
    df_cleaned = Cleaner(df)
    
    # 2. Get your columns and types instantly printed out
    colname(df_cleaned)
    
    # 3. Use any column name from the list to get its dedicated plot!
    print("Generating single-column plots...")
    plot(df_cleaned, "survived")      # Renders Category Composition (binary discrete -> categorical)
    plot(df_cleaned, "age")           # Renders lightweight Box & Histogram layout (numeric)
    plot(df_cleaned, "ticket")        # Renders Category Composition (high cardinality string -> categorical)
    
    # 4. Generate the universal subplot matrix of all column combinations!
    print("\nGenerating pairwise combination matrix subplot grid...")
    # Plots the first 4 columns by default to make it look clean
    plot(df_cleaned, columns=["survived", "pclass", "age", "fare"])
    
    # 5. Generate and open the premium interactive HTML dashboard!
    print("\nGenerating interactive HTML dashboard...")
    plot_dashboard(df_cleaned, filename="ezclean_dashboard.html")

if __name__ == "__main__":
    main()