from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from typing import Dict, Any

class CreditModeler:
    """Class for building and evaluating machine learning models on German Credit data."""
    
    def __init__(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, 
                 random_state: int = 42):
        """Initialize with feature matrix and target variable.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
        """
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.scaler = None
        self.model = None
        
    def split_data(self) -> None:
        """Split data into training and test sets and scale numerical features."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state,
            stratify=self.y
        )
        
        # Scale numerical features
        numerical_cols = self.X.select_dtypes(include=['int64', 'float64']).columns
        if len(numerical_cols) > 0:
            self.scaler = StandardScaler()
            self.X_train[numerical_cols] = self.scaler.fit_transform(self.X_train[numerical_cols])
            self.X_test[numerical_cols] = self.scaler.transform(self.X_test[numerical_cols])
        
        assert len(self.X_train) > 0 and len(self.X_test) > 0, "Data splitting failed"
    
    def train_model(self, **kwargs: Any) -> None:
        """Train a Random Forest classifier on the training data.
        
        Args:
            **kwargs: Additional arguments to pass to RandomForestClassifier
        """
        assert self.X_train is not None, "Data not split - call split_data() first"
        
        # Set default parameters if not provided
        params = {
            'n_estimators': 100,
            'random_state': self.random_state,
            'class_weight': 'balanced'
        }
        params.update(kwargs)
        
        self.model = RandomForestClassifier(**params)
        self.model.fit(self.X_train, self.y_train)
        
        assert self.model is not None, "Model training failed"
    
    def evaluate_model(self) -> Dict:
        """Evaluate the trained model on test data.
        
        Returns:
            Dict: Dictionary containing evaluation metrics
        """
        assert self.model is not None, "Model not trained - call train_model() first"
        assert self.X_test is not None, "Data not split - call split_data() first"
        
        y_pred = self.model.predict(self.X_test)
        
        return {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'classification_report': classification_report(self.y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist()
        }
    
    def get_feature_importances(self) -> pd.Series:
        """Get feature importances from the trained model.
        
        Returns:
            pd.Series: Feature importances sorted in descending order
        """
        assert self.model is not None, "Model not trained - call train_model() first"
        return pd.Series(
            self.model.feature_importances_,
            index=self.X.columns
        ).sort_values(ascending=False)