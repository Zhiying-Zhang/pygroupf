import pandas as pd
import os


def load_data(file_path):
    """
    Load the dataset from the specified file path and perform initial formatting.

    Args:
        file_path (str): Path to the CSV file containing the data.

    Returns:
        pd.DataFrame: The raw loaded data with standardized column names.
    """
    data = pd.read_csv(file_path)

    # Remove unnamed index column if exists
    if data.columns[0].startswith("Unnamed"):
        data = data.drop(data.columns[0], axis=1)

    # Standardize column names
    data.columns = data.columns.str.lower().str.replace(" ", "_")

    assert not data.empty, "Loaded DataFrame should not be empty"
    return data


def clean_data(data, categorical_cols, numerical_cols):
    """
    Clean and preprocess the dataset.

    Handles missing values differently for categorical and numerical columns:
    - Categorical columns: Filled with 'unknown'
    - Numerical columns: Converted to numeric type and filled with median

    Args:
        data (pd.DataFrame): The DataFrame to be cleaned.
        categorical_cols (list): List of column names to be treated as categorical.
        numerical_cols (list): List of column names to be treated as numerical.

    Returns:
        pd.DataFrame: The cleaned DataFrame with no missing values.
    """

    # Convert user-provided column names to internal format
    categorical_cols = [col.lower().replace(" ", "_") for col in categorical_cols]
    numerical_cols = [col.lower().replace(" ", "_") for col in numerical_cols]

    # Process categorical columns
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].fillna("unknown")
            assert data[col].isna().sum() == 0, f"{col} still contains NA values"

    # Process numerical columns
    for col in numerical_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
            median_val = data[col].median()
            data[col] = data[col].fillna(median_val)
            assert data[col].isna().sum() == 0, f"{col} still contains NA values"
            assert pd.api.types.is_numeric_dtype(
                data[col]
            ), f"{col} is not numeric after conversion"

    return data


class DataEncoder:
    """
    A class for encoding categorical columns in a DataFrame using specified numerical mappings.

    Attributes:
        column_mapping (dict): Dictionary mapping column names to value dictionaries.
    """

    def __init__(self, column_mapping):
        """
        Initialize the CategoricalEncoder with the column mapping.

        Args:
            column_mapping (dict): Dictionary mapping column names to value dictionaries.
                                 Example: {'sex': {'male': 1, 'female': 0}}
        """
        self.column_mapping = column_mapping

    def encode(self, data):
        """
        Encode categorical columns in the DataFrame using the specified numerical mappings.

        Args:
            data (pd.DataFrame): The DataFrame containing categorical columns to be encoded.

        Returns:
            pd.DataFrame: The DataFrame with encoded categorical values.
        """

        # Convert the mapping keys to internal column name format
        internal_mapping = {}
        for col, value_map in self.column_mapping.items():
            internal_col = col.lower().replace(" ", "_")
            internal_mapping[internal_col] = value_map

        # Apply the mapping to each column
        for col, value_map in internal_mapping.items():
            if col in data.columns:
                data[col] = data[col].map(value_map)

        return data


def save_processed_data(data, output_dir="data", filename="processed_credit_data.csv"):
    """
    Save the processed data to a CSV file.

    Args:
        data (pd.DataFrame): The DataFrame to be saved.
        output_dir (str): Directory to save the processed data.
        filename (str): Name of the output file.

    Returns:
        str: Path where the file was saved.
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save processed data to CSV
    output_path = os.path.join(output_dir, filename)
    data.to_csv(output_path, index=False)
    print(f"Data processing completed successfully! File saved to {output_path}")

    return output_path
