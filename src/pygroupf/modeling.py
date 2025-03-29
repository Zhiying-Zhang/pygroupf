"""
Random Forest Classifier for Credit Data

This script reads processed credit data, applies one-hot encoding,
creates a target column, splits the data into training and test sets,
performs hyperparameter tuning using GridSearchCV, trains a Random Forest model,
and evaluates the model's performance using accuracy, confusion matrix, and F2-score.
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, fbeta_score

# Read data
data = pd.read_csv("data/processed_credit_data.csv")
print("Columns in dataset:", data.columns)  # Print column names for verification

# One-Hot Encoding of Categorical Features
categorical_columns = ['sex', 'housing', 'saving_accounts', 'checking_account', 'purpose']
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)  # Prevent dummy variable trap

# Creating Target Column
def create_target_column(df):
    """Creates a binary target column 'good_credit' based on credit amount and age."""
    df['good_credit'] = ((df['credit_amount'] > 10000) & (df['age'] > 30)).astype(int)
    return df

data = create_target_column(data)

# Splitting dataset into features (X) and target (y)
X = data.drop(columns=['good_credit'])  # Feature matrix
y = data['good_credit']  # Target column

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Setting the Hyperparameter Grid
param_grid = {
    "max_depth": [3, 5, 7, 10, None],
    "n_estimators": [3, 5, 10, 25, 50, 150],
    "max_features": [4, 7, 15, 20]
}

# Creating a Random Forest Classifier
model = RandomForestClassifier(random_state=2)

# Hyperparameter tuning using GridSearchCV
def tune_hyperparameters(model, param_grid, X_train, y_train):
    """Performs hyperparameter tuning using GridSearchCV and returns the best model."""
    grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='recall', verbose=4)
    grid_search.fit(X_train, y_train)
    print("Best score:", grid_search.best_score_)
    print("Best params:", grid_search.best_params_)
    return grid_search.best_params_

best_params = tune_hyperparameters(model, param_grid, X_train, y_train)

# Training Random Forest Model with Optimal Parameters
def train_model(X_train, y_train, best_params):
    """Trains a Random Forest model using the best hyperparameters."""
    rf = RandomForestClassifier(**best_params, random_state=2)
    rf.fit(X_train, y_train)
    return rf

rf = train_model(X_train, y_train, best_params)

# Making Predictions
y_pred = rf.predict(X_test)

# Model Evaluation
def evaluate_model(y_test, y_pred):
    """Evaluates the trained model using accuracy, confusion matrix, and F2-score."""
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("F2 Score:", fbeta_score(y_test, y_pred, beta=2))

evaluate_model(y_test, y_pred)
