import pandas as pd
from typing import Tuple

class DataProcessor:
    """Class for loading and preprocessing German Credit data.
    
    Args:
        data_path (str): Path to the CSV file containing the data.
    
    Attributes:
        data (pd.DataFrame): Raw data loaded from the CSV.
        X (pd.DataFrame): Feature matrix after preprocessing.
        y (pd.Series): Target variable after preprocessing.
    """
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        
    def load_data(self):
        """Load data from CSV file."""
        self.data = pd.read_csv(self.data_path)
        assert not self.data.empty, "Data loading failed - DataFrame is empty"
        
    def preprocess(self):
        """Preprocess the data by handling missing values and encoding categorical variables."""
        assert self.data is not None, "Data not loaded - call load_data() first"
        
        # Convert target to binary (1 = Good, 0 = Bad)
        self.data['Risk'] = self.data['Risk'].map({'good': 1, 'bad': 0})
        self.y = self.data['Risk']
        
        # Drop the target and unnecessary columns
        self.X = self.data.drop(['Risk', 'Unnamed: 0'], axis=1, errors='ignore')
        
        # Convert categorical variables to dummy variables
        self.X = pd.get_dummies(self.X, drop_first=True)
        
        assert not self.X.isnull().any().any(), "Missing values detected after preprocessing"
        assert len(self.X) == len(self.y), "X and y have different lengths after preprocessing"
    
    def get_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Get the processed feature matrix and target variable.
        
        Returns:
            Tuple containing:
                - X (pd.DataFrame): Feature matrix
                - y (pd.Series): Target variable
        """
        assert self.X is not None and self.y is not None, "Data not processed - call preprocess() first"
        return self.X, self.y
    
