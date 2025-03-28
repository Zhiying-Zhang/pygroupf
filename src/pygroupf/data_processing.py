
import pandas as pd
import os


class DataProcessor:
    """
    A class for processing and cleaning the German credit dataset.

    This class handles data loading, cleaning, categorical encoding, and saving processed data.
    It ensures consistent column naming, handles missing values, and validates data integrity.

    Attributes:
        file_path (str): Path to the CSV file containing the raw data.
        data (pd.DataFrame): The loaded and processed DataFrame after method calls.
    """

    def __init__(self, file_path):
        """
        Initialize the DataProcessor with the path to the raw data file.

        Args:
            file_path (str): Path to the CSV file containing the German credit data.
                            Expected to be a valid path to a readable CSV file.
        """
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """
        Load the dataset from the specified file path and perform initial formatting.

        Performs the following operations:
        1. Reads the CSV file into a DataFrame
        2. Removes any unnamed index columns
        3. Standardizes column names to lowercase with underscores

        Returns:
            pd.DataFrame: The raw loaded data with standardized column names.

        Raises:
            FileNotFoundError: If the specified file path doesn't exist.
            ValueError: If the file cannot be parsed as CSV.
            AssertionError: If the loaded DataFrame is empty.
        """
        self.data = pd.read_csv(self.file_path)

        # Remove unnamed index column if present
        if self.data.columns[0].startswith("Unnamed"):
            self.data = self.data.drop(self.data.columns[0], axis=1)

        # Standardize column names
        self.data.columns = self.data.columns.str.lower().str.replace(" ", "_")

        assert not self.data.empty, "Loaded DataFrame should not be empty"
        return self.data

    def clean_data(self, categorical_cols, numerical_cols):
        """
        Clean and preprocess the loaded dataset.

        Handles missing values differently for categorical and numerical columns:
        - Categorical columns: Filled with 'unknown'
        - Numerical columns: Converted to numeric type and filled with median

        Args:
            categorical_cols (list): List of column names to be treated as categorical.
            numerical_cols (list): List of column names to be treated as numerical.

        Returns:
            pd.DataFrame: The cleaned DataFrame with no missing values.

        Raises:
            ValueError: If data hasn't been loaded first.
            AssertionError: If cleaning produces invalid results (remaining NA values
                          or incorrect column types).
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Convert user-provided column names to internal format
        categorical_cols = [col.lower().replace(" ", "_") for col in categorical_cols]
        numerical_cols = [col.lower().replace(" ", "_") for col in numerical_cols]

        # Process categorical columns
        for col in categorical_cols:
            if col in self.data.columns:
                self.data[col] = self.data[col].fillna("unknown")
                assert self.data[col].isna().sum() == 0, f"{col} still contains NA values"

        # Process numerical columns
        for col in numerical_cols:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors="coerce")
                median_val = self.data[col].median()
                self.data[col] = self.data[col].fillna(median_val)
                assert self.data[col].isna().sum() == 0, f"{col} still contains NA values"
                assert pd.api.types.is_numeric_dtype(self.data[col]), f"{col} is not numeric after conversion"
        
        return self.data

    def encode_categorical_values(self, column_mapping):
        """
        Encode categorical columns using specified numerical mappings.

        Typical usage would include encoding columns like:
        - sex: {'male': 1, 'female': 0}
        - housing: {'own': 2, 'free': 1, 'rent': 0}
        - saving_accounts: {'unknown': 0, 'little': 1, ..., 'rich': 4}

        Args:
            column_mapping (dict): Dictionary mapping column names to value dictionaries.
                                 Example: {'sex': {'male': 1, 'female': 0}}

        Returns:
            pd.DataFrame: The DataFrame with encoded categorical values.

        Raises:
            ValueError: If data hasn't been loaded first.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Convert the mapping keys to internal column name format
        internal_mapping = {}
        for col, value_map in column_mapping.items():
            internal_col = col.lower().replace(" ", "_")
            internal_mapping[internal_col] = value_map

        # Apply the mapping to each column
        for col, value_map in internal_mapping.items():
            if col in self.data.columns:
                self.data[col] = self.data[col].map(value_map)
        
        return self.data

    def get_processed_data(self):
        """
        Retrieve the fully processed data and save it to a CSV file.

        The processed data is saved to 'data/processed_credit_data.csv'. The 'data'
        directory will be created if it doesn't exist.

        Returns:
            pd.DataFrame: The fully processed DataFrame ready for analysis.

        Raises:
            ValueError: If data hasn't been processed yet (clean_data() not called).
        """
        if self.data is None:
            raise ValueError("Data not processed. Call clean_data() first.")

        # Create data folder if it doesn't exist
        os.makedirs("data", exist_ok=True)

        # Save processed data to CSV
        output_path = os.path.join("data", "processed_credit_data.csv")
        self.data.to_csv(output_path, index=False)
        print("Data processing completed successfully!")

        return self.data      
    

# # Define column types
# categorical_cols = [
#     "Sex",
#     "Housing",
#     "Saving accounts",
#     "Checking account",
#     "Purpose",
# ]
# numerical_cols = ["Age", "Job", "Credit amount", "Duration"]

# """
# Encode specific categorical columns with numerical values as per requirements:
# - sex: male=1, female=0
# - housing: own=2, free=1, rent=0
# - saving_accounts: unknown=0, little=1, moderate=2, quite rich=3, rich=4
# - checking_account: unknown=0, little=1, moderate=2, rich=3
# """

# # Define mapping of categorical values to numerical values
# mapping = {
#     "Sex": {"male": 1, "female": 0, "unknown": -1},
#     "Housing": {"own": 2, "free": 1, "rent": 0, "unknown": -1},
#     "Saving accounts": {
#         "unknown": 0,
#         "little": 1,
#         "moderate": 2,
#         "quite rich": 3,
#         "rich": 4,
#     },
#     "Checking account": {"unknown": 0, "little": 1, "moderate": 2, "rich": 3},
# }


# # data processing
# processor = DataProcessor("data/german_credit_data.csv")
# processor.load_data()
# processor.clean_data(categorical_cols, numerical_cols)
# encode_data = processor.encode_categorical_values(mapping)
# processed_data = processor.get_processed_data()