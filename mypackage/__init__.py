from .data_processing import DataProcessor
from .analysis import DataAnalyzer
from .visualization import DataVisualizer

__all__ = ["DataProcessor", "DataAnalyzer", "DataVisualizer"]

import mypackage

processor = mypackage.DataProcessor("german_credit_data.csv")
analyzer = mypackage.DataAnalyzer("german_credit_data.csv")
visualizer = mypackage.DataVisualizer("german_credit_data.csv")
