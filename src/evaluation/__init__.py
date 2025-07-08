"""
Evaluation module for Carbon Emissions Forecasting System v3.0.

This module contains model evaluation and performance assessment
functionality following IEEE standards.

Author: Kelsi Naidoo
Institution: University of Cape Town
"""

from .metrics import ModelEvaluator
from .visualization import ResultsVisualizer

__all__ = ['ModelEvaluator', 'ResultsVisualizer'] 