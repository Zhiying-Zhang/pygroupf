from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd

class RiskPredictor:
    def __init__(self, data):
        self.data = data
        self.model = RandomForestClassifier()
        self.encoder = LabelEncoder()

    def preprocess_data(self):
        # 编码分类变量
        categorical_cols = ["Sex", "Housing", "Purpose", "Saving accounts", "Checking account"]
        for col in categorical_cols:
            self.data[col] = self.encoder.fit_transform(self.data[col])
        
        X = self.data[["Age", "Sex", "Job", "Credit amount", "Duration", "Purpose"]]
        y = self.data["risk"]
        return X, y

    def train_model(self):
        X, y = self.preprocess_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        print(f"Model Accuracy: {accuracy_score(y_test, y_pred)}")

    def predict_risk(self, new_data):
        return self.model.predict(new_data)