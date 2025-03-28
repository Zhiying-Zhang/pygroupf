import pandas as pd  # Import pandas for data manipulation
import os  # Import os to handle file paths and directories


class DataProcessor:
    """
    A class for processing and cleaning the German credit dataset.
    """

    def __init__(self, file_path):
        """
        Initialize the DataProcessor with the path to the raw data file.
        """
        # Define the path for the raw data file
        self.file_path = "D:/Personal/study/CS/Python Seminar/github clone/pygroupf/data/german_credit_data.csv"
        # Store the file path
        self.data = "D:/Personal/study/CS/Python Seminar/github clone/pygroupf/data/german_credit_data.csv"

    def load_data(self):
        """
        Load the dataset from the specified file path and perform initial formatting.
        """
        # Check if the file exists at the specified path
        if not os.path.exists(self.file_path):
            # If the file does not exist, raise an error
            raise FileNotFoundError(f"The file at {self.file_path} does not exist.")
        
        print("Loading data from:", self.file_path)
        # Load the data from the file into a pandas DataFrame
        self.data = pd.read_csv(self.file_path)

        # Remove unnamed index column if present (usually the first column)
        if self.data.columns[0].startswith("Unnamed"):
            self.data = self.data.drop(self.data.columns[0], axis=1)

        # Standardize column names by converting them to lowercase and replacing spaces with underscores
        self.data.columns = self.data.columns.str.lower().str.replace(r"\s+", "_", regex=True)

        # Ensure the data is not empty
        assert not self.data.empty, "Loaded DataFrame should not be empty"
        print("Data loaded successfully.")
        return self.data

    def clean_data(self, categorical_cols, numerical_cols):
        """
        Clean and preprocess the loaded dataset.
        """
        # Check if the data has been loaded before cleaning
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("Cleaning data...")

        # Convert user-provided column names to internal format (lowercase and underscores instead of spaces)
        categorical_cols = [col.lower().replace(" ", "_") for col in categorical_cols]
        numerical_cols = [col.lower().replace(" ", "_") for col in numerical_cols]

        # Process categorical columns by filling missing values with 'unknown'
        for col in categorical_cols:
            if col in self.data.columns:
                self.data[col] = self.data[col].fillna("unknown")
                # Ensure no missing values remain in categorical columns
                assert self.data[col].isna().sum() == 0, f"{col} still contains NA values"

        # Process numerical columns by converting to numeric and filling missing values with the median
        for col in numerical_cols:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors="coerce")
                # Use median value to fill missing numerical values
                median_val = self.data[col].median()
                self.data[col] = self.data[col].fillna(median_val)
                # Ensure no missing values remain in numerical columns
                assert self.data[col].isna().sum() == 0, f"{col} still contains NA values"
                # Check that the column is numeric after conversion
                assert pd.api.types.is_numeric_dtype(self.data[col]), f"{col} is not numeric after conversion"
        
        print("Data cleaned successfully.")
        return self.data

    def encode_categorical_values(self, column_mapping):
        """
        Encode categorical columns using specified numerical mappings.
        """
        # Check if the data has been loaded before encoding
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        print("Encoding categorical values...")
        internal_mapping = {}
        # Create internal mapping with standardized column names
        for col, value_map in column_mapping.items():
            internal_col = col.lower().replace(" ", "_")
            internal_mapping[internal_col] = value_map

        # Apply the mapping to each column
        for col, value_map in internal_mapping.items():
            if col in self.data.columns:
                self.data[col] = self.data[col].map(value_map)
        
        print("Categorical values encoded successfully.")
        return self.data

    def get_processed_data(self):
        """
        Retrieve the fully processed data and save it to a CSV file.
        """
        # Check if the data has been processed before retrieving
        if self.data is None:
            raise ValueError("Data not processed. Call clean_data() first.")

        # Create data folder if it doesn't exist
        os.makedirs("D:/Personal/study/CS/Python Seminar/github clone/pygroupf/data", exist_ok=True)

        # Define the output path for the processed data file
        output_path = os.path.join("D:/Personal/study/CS/Python Seminar/github clone/pygroupf/data", "processed_credit_data.csv")
        # Save the DataFrame to a CSV file
        self.data.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}.")

        return self.data


if __name__ == "__main__":
    # Set the file path for the raw data
    file_path = "D:/Personal/study/CS/Python Seminar/github clone/pygroupf/data"  
    processor = DataProcessor(file_path)

    # Load the data
    processor.load_data()

    # Clean the data (provide actual column names for categorical and numerical columns)
    categorical_cols = ['sex', 'housing']  # List of categorical column names
    numerical_cols = ['age', 'credit_amount']  # List of numerical column names
    processor.clean_data(categorical_cols, numerical_cols)

    # Encode categorical values (provide actual mappings for encoding)
    column_mapping = {'sex': {'male': 1, 'female': 0}}  # Example of encoding categorical columns
    processor.encode_categorical_values(column_mapping)

    # Get the processed data and save it to the specified folder
    processor.get_processed_data()
