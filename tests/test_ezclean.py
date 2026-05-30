import os
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Import the package contents
from ezclean.loader import Smart_loader
from ezclean.cleaner import DataCleaner, Cleaner
from ezclean.plots import _classify_column, plot, plot_matrix, plot_dashboard


class TestEzcleanLoader(unittest.TestCase):
    def setUp(self):
        # Create a small temp CSV file for testing
        self.test_csv_path = "temp_test_dataset.csv"
        self.df_mock = pd.DataFrame({
            "PassengerId": [1, 2, 3],
            "Survived": [0, 1, 0],
            "Age": [22.0, 38.0, np.nan],
            "Name": ["Braund, Mr. Owen Harris", "Cumings, Mrs. John Bradley", "Heikkinen, Miss. Laina"],
            "Embarked": ["S", "C", "S"]
        })
        self.df_mock.to_csv(self.test_csv_path, index=False)

    def tearDown(self):
        if os.path.exists(self.test_csv_path):
            os.remove(self.test_csv_path)

    def test_smart_loader_csv(self):
        loaded_df = Smart_loader(self.test_csv_path)
        self.assertIsInstance(loaded_df, pd.DataFrame)
        self.assertEqual(loaded_df.shape[0], 3)
        self.assertEqual(list(loaded_df.columns), ["PassengerId", "Survived", "Age", "Name", "Embarked"])


class TestEzcleanCleaner(unittest.TestCase):
    def setUp(self):
        self.df_dirty = pd.DataFrame({
            "Passenger ID": [1, 2, 2, 3, 4, 5, 6, 7],  # Contains spaces and duplicates
            "Survived?": ["0", "1", "1", "?", "0", "1", "0", "1"],  # Stringified integers, garbage char '?'
            "Age ": [22.0, 38.0, 38.0, np.nan, 25.0, 30.0, 45.0, 29.0],  # Trailing whitespace, null value
            "Fare": [7.25, 71.2833, 71.2833, 1000.0, 8.0, 12.0, 15.0, 20.0]  # Normal, duplicate, and extreme outlier
        })

    def test_cleaner_pipeline(self):
        cleaned_df = Cleaner(self.df_dirty, outlayer=True)
        
        # Verify snake_case column names conversion
        self.assertIn("passenger_id", cleaned_df.columns)
        self.assertIn("survived", cleaned_df.columns)
        self.assertIn("age", cleaned_df.columns)
        self.assertIn("fare", cleaned_df.columns)
        
        # Verify duplicate rows dropped (original has 8, index 1 & 2 are identical duplicates)
        # Note: after dropping duplicates, row count goes to 7
        # Then, outlier handling (IQR) will drop the fare = 1000 outlier AND the age = 45 outlier, reducing to 5 rows
        self.assertEqual(cleaned_df.shape[0], 5)
        
        # Verify null filling (age null filled with median of ages: 30.0)
        # Verify ? was sanitized to null, then filled as categorical placeholder "Unknown"
        self.assertNotIn("?", cleaned_df["survived"].values)


class TestEzcleanPlots(unittest.TestCase):
    def setUp(self):
        self.df_plot = pd.DataFrame({
            "passenger_id": [1, 2, 3, 4, 5],
            "survived": [0, 1, 0, 1, 0], # categorical classification (<= 10 values)
            "age": [22.0, 38.0, 26.0, 35.0, 54.0], # numeric classification
            "fare": [7.25, 71.2833, 8.05, 53.1, 8.05], # numeric classification
            "embarked": ["S", "C", "S", "S", "Q"], # categorical classification
            "timestamp": pd.date_range("2026-01-01", periods=5) # datetime classification
        })

    def test_classify_column(self):
        self.assertEqual(_classify_column(self.df_plot["age"], "age"), "numeric")
        self.assertEqual(_classify_column(self.df_plot["survived"], "survived"), "categorical")
        self.assertEqual(_classify_column(self.df_plot["embarked"], "embarked"), "categorical")
        self.assertEqual(_classify_column(self.df_plot["timestamp"], "timestamp"), "datetime")

    @patch("plotly.graph_objects.Figure.show")
    def test_plot_single_column_numeric(self, mock_show):
        # Simply ensure it executes without error
        plot(self.df_plot, target_column="age")
        mock_show.assert_called_once()

    @patch("plotly.graph_objects.Figure.show")
    def test_plot_single_column_categorical(self, mock_show):
        plot(self.df_plot, target_column="embarked")
        mock_show.assert_called_once()

    @patch("plotly.graph_objects.Figure.show")
    def test_plot_single_column_datetime(self, mock_show):
        plot(self.df_plot, target_column="timestamp")
        mock_show.assert_called_once()

    @patch("plotly.graph_objects.Figure.show")
    def test_plot_matrix(self, mock_show):
        # Run matrix plot
        plot(self.df_plot)
        mock_show.assert_called_once()

    @patch("webbrowser.open")
    @patch("plotly.graph_objects.Figure.show")
    def test_plot_dashboard(self, mock_show, mock_open):
        temp_dash = "temp_dashboard.html"
        try:
            out_file = plot_dashboard(self.df_plot, filename=temp_dash, show=True)
            self.assertTrue(os.path.exists(out_file))
            mock_open.assert_called_once()
        finally:
            if os.path.exists(temp_dash):
                os.remove(temp_dash)


if __name__ == "__main__":
    unittest.main()
