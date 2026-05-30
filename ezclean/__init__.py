# ezclean/__init__.py

from .loader import Smart_loader
from .cleaner import Cleaner
from .plots import colname, plot, plot_matrix, plot_dashboard

__all__ = ['Smart_loader', 'Cleaner', 'colname', 'plot', 'plot_matrix', 'plot_dashboard']