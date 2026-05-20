import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

class Visualizer:
    @staticmethod
    def plot_horizontal_violin(df, x, y=None, color=None, title="Horizontal Violin Plot"):
        """
        Interactive Horizontal violin plots powered by Plotly.
        x should be the numeric column, y can be categorical.
        """
        fig = px.violin(df, x=x, y=y, color=color, orientation='h', box=True, points="all", title=title)
        fig.update_layout(template="plotly_dark", title_x=0.5)
        return fig

    @staticmethod
    def plot_scatter(df, x, y, color=None, size=None, title="Scatter Plot"):
        """
        Interactive scatter plots.
        """
        fig = px.scatter(df, x=x, y=y, color=color, size=size, title=title)
        fig.update_layout(template="plotly_dark", title_x=0.5)
        return fig

    @staticmethod
    def plot_histogram(df, x, color=None, nbins=None, title="Histogram"):
        """
        Interactive histograms.
        """
        fig = px.histogram(df, x=x, color=color, nbins=nbins, marginal="box", title=title)
        fig.update_layout(template="plotly_dark", title_x=0.5, barmode='overlay')
        fig.update_traces(opacity=0.75)
        return fig

    @staticmethod
    def plot_grouped_bar(df, x, y, color=None, barmode='group', title="Grouped Bar Chart"):
        """
        Interactive grouped bar charts.
        """
        fig = px.bar(df, x=x, y=y, color=color, barmode=barmode, title=title)
        fig.update_layout(template="plotly_dark", title_x=0.5)
        return fig
