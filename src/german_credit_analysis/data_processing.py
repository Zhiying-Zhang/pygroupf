import pandas as pd
import os

class DataProcessor:
    """
    A class to process the German credit dataset.

    Attributes:
        file_path (str): Path to the CSV file containing the data
        data (pd.DataFrame): The loaded and processed DataFrame
    """

    def __init__(self, file_path):
        """
        Initialize the DataProcessor with a file path.

        Args:
            file_path: Path to the CSV file containing the German credit data
        """    
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """
        Load the data from the specified file path.

        Returns:
            pd.DataFrame: The raw loaded data

        Raises:
            FileNotFoundError: If the specified file path doesn't exist
            ValueError: If the file cannot be parsed as CSV
        """
        self.data = pd.read_csv(self.file_path)

        # Remove unnamed index column if present
        if self.data.columns[0].startswith('Unnamed'):
            self.data = self.data.drop(self.data.columns[0], axis=1)

        # Add customer_id column if it doesn't exist
        if 'customer_id' not in self.data.columns:
            self.data.insert(0, 'customer_id', range(1, len(self.data) + 1))

        # Standardize column names
        self.data.columns = self.data.columns.str.lower().str.replace(' ', '_')

        assert not self.data.empty, "Loaded DataFrame should not be empty"
        return self.data
    
    def clean_data(self):
        """
        Clean and preprocess the loaded data.

        - Numerical columns: Fill missing values with median
        - Categorical columns: Fill missing values with 'unknown'
        - Validate column types after cleaning

        Returns:
            pd.DataFrame: The cleaned DataFrame

        Raises:
            ValueError: If data hasn't been loaded first
            AssertionError: If cleaning produces invalid results
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Define column types
        categorical_cols = ["sex", "housing", "saving_accounts", "checking_account", "purpose"]
        numerical_cols = ["age", "job", "credit_amount", "duration"]

        # Process categorical columns and fill missing values with 'unknown'
        for col in categorical_cols:
            if col in self.data.columns:
                self.data[col] = self.data[col].fillna("unknown")
                assert self.data[col].isna().sum() == 0, f"{col} still contains NA values"

        # Process numerical columns and fill missing values with median
        for col in numerical_cols:
            if col in self.data.columns:
                # Convert to numeric, coercing errors to NaN
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                median_val = self.data[col].median()
                self.data[col] = self.data[col].fillna(median_val)
                assert self.data[col].isna().sum() == 0, f"{col} still contains NA values"
                assert pd.api.types.is_numeric_dtype(self.data[col]), f"{col} is not numeric after conversion"
        
        return self.data
    
    def encode_categorical_values(self):
        """
        Encode specific categorical columns with numerical values as per requirements:
        - sex: male=1, female=0
        - housing: own=2, free=1, rent=0
        - saving_accounts: unknown=0, little=1, moderate=2, quite rich=3, rich=4
        - checking_account: unknown=0, little=1, moderate=2, rich=3
        
        Returns:
            pd.DataFrame: The DataFrame with encoded categorical values
            
        Raises:
            ValueError: If data hasn't been cleaned first
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        # Sex encoding
        if "sex" in self.data.columns:
            self.data["sex"] = self.data["sex"].map({"male": 1, "female": 0, "unknown": -1})
            
        # Housing encoding
        if "housing" in self.data.columns:
            self.data["housing"] = self.data["housing"].map({"own": 2, "free": 1, "rent": 0, "unknown": -1})
            
        # Saving accounts encoding
        if "saving_accounts" in self.data.columns:
            self.data["saving_accounts"] = self.data["saving_accounts"].map({
                "none": 0,
                "little": 1,
                "moderate": 2,
                "quite rich": 3,
                "rich": 4,
                "unknown": -1
            })
            
        # Checking account encoding
        if "checking_account" in self.data.columns:
            self.data["checking_account"] = self.data["checking_account"].map({
                "none": 0,
                "little": 1,
                "moderate": 2,
                "rich": 3,
                "unknown": -1
            })
            
        return self.data

    def get_processed_data(self):
        """
        Get the processed data after cleaning and save it to a CSV file in the data folder.

        Returns:
            pd.DataFrame: The processed DataFrame

        Raises:
            ValueError: If data hasn't been processed yet
        """
        if self.data is None:
            raise ValueError("Data not processed. Call clean_data() first.")
        
        # Create data folder if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Save processed data to CSV
        output_path = os.path.join('data', 'processed_credit_data.csv')
        self.data.to_csv(output_path, index=False)
        
        return self.data