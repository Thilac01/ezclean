import pandas as pd
import numpy as np
import scipy.stats as ss
import plotly.express as px
import plotly.graph_objects as go

class StatInsights:
    @staticmethod
    def _cramers_v(x, y):
        confusion_matrix = pd.crosstab(x, y)
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        if n <= 1:
            return 0.0
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        
        # Handle cases where denominator is 0
        denominator = min((kcorr - 1), (rcorr - 1))
        if denominator == 0:
            return 0.0
            
        return np.sqrt(phi2corr / denominator)

    @staticmethod
    def _correlation_ratio(categories, measurements):
        fcat, _ = pd.factorize(categories)
        cat_num = np.max(fcat) + 1
        y_avg_array = np.zeros(cat_num)
        n_array = np.zeros(cat_num)
        for i in range(0, cat_num):
            cat_measures = measurements[np.argwhere(fcat == i).flatten()]
            n_array[i] = len(cat_measures)
            if n_array[i] > 0:
                y_avg_array[i] = np.average(cat_measures)
            
        y_total_avg = np.average(measurements)
        numerator = np.sum(n_array * np.square(y_avg_array - y_total_avg))
        denominator = np.sum(np.square(measurements - y_total_avg))
        
        if denominator == 0:
            return 0.0
        return np.sqrt(numerator / denominator)

    @staticmethod
    def pearson_heatmap(df, title="Pearson Correlation Heatmap (Numeric)"):
        """
        Pearson Correlation heatmaps for numeric data.
        """
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            raise ValueError("No numeric columns found for Pearson correlation.")
            
        corr_matrix = numeric_df.corr(method='pearson')
        fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', title=title)
        fig.update_layout(template="plotly_dark", title_x=0.5)
        return fig

    @staticmethod
    def cramers_v_heatmap(df, title="Cramér's V Heatmap (Categorical)"):
        """
        Cramér's V heatmaps for categorical associations.
        """
        cat_df = df.select_dtypes(include=['object', 'category'])
        if cat_df.empty:
            raise ValueError("No categorical columns found for Cramér's V correlation.")
            
        cols = cat_df.columns
        corr_matrix = pd.DataFrame(np.ones((len(cols), len(cols))), index=cols, columns=cols)
        
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                corr = StatInsights._cramers_v(cat_df[cols[i]], cat_df[cols[j]])
                corr_matrix.loc[cols[i], cols[j]] = corr
                corr_matrix.loc[cols[j], cols[i]] = corr
                
        fig = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale='Blues', title=title)
        fig.update_layout(template="plotly_dark", title_x=0.5)
        return fig

    @staticmethod
    def unified_association_heatmap(df, title="Unified Association Heatmap"):
        """
        Unified Association Heatmaps combining Numeric and Categorical data.
        (using Point-Biserial and Eta/ANOVA via correlation ratio).
        """
        cols = df.columns
        corr_matrix = pd.DataFrame(np.ones((len(cols), len(cols))), index=cols, columns=cols)
        
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                col1, col2 = cols[i], cols[j]
                is_num1 = pd.api.types.is_numeric_dtype(df[col1])
                is_num2 = pd.api.types.is_numeric_dtype(df[col2])
                
                # Both Numeric -> Pearson
                if is_num1 and is_num2:
                    corr = df[[col1, col2]].corr().iloc[0, 1]
                # Both Categorical -> Cramer's V
                elif not is_num1 and not is_num2:
                    corr = StatInsights._cramers_v(df[col1], df[col2])
                # Mixed -> Correlation Ratio (Eta)
                else:
                    if is_num1:
                        corr = StatInsights._correlation_ratio(df[col2], df[col1])
                    else:
                        corr = StatInsights._correlation_ratio(df[col1], df[col2])
                        
                # Take absolute value for unified metric 0 to 1 scale roughly
                corr_matrix.loc[col1, col2] = abs(corr)
                corr_matrix.loc[col2, col1] = abs(corr)
                
        fig = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale='Viridis', title=title)
        fig.update_layout(template="plotly_dark", title_x=0.5)
        return fig
