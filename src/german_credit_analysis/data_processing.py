import pandas as pd

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        return self.data

    def clean_data(self):
        # 处理缺失值（示例：填充分类列的NA为"unknown"，数值列填充为0）
        categorical_cols = ["Saving accounts", "Checking account", "Purpose"]
        numerical_cols = ["Credit amount", "Duration"]
        
        for col in categorical_cols:
            self.data[col] = self.data[col].fillna("unknown")
        
        for col in numerical_cols:
            self.data[col] = self.data[col].fillna(0)
        
        return self.data

    def get_processed_data(self):
        return self.data