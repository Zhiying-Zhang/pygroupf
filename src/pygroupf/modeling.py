import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, fbeta_score

# read data
data = pd.read_csv("data/processed_credit_data.csv")

# Prints the column names and checks that they match the expected column names
print(data.columns)

# One-Hot Encoding of Classification Features
categorical_columns = ['sex', 'housing', 'saving_accounts', 'checking_account', 'purpose']  # taxonomic listing
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)  # drop_first=True Preventing dummy variable traps

# Creating Target Columns
data['good_credit'] = ((data['credit_amount'] > 10000) & (data['age'] > 30)).astype(int)

# Feature column X and target column y
X = data.drop(columns=['good_credit'])  # Feature column
y = data['good_credit']  # target column

# Slicing the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Setting the Hyperparameter Grid
param_grid = {
    "max_depth": [3, 5, 7, 10, None],
    "n_estimators": [3, 5, 10, 25, 50, 150],
    "max_features": [4, 7, 15, 20]
}

# Creating a Random Forest Classifier
model = RandomForestClassifier(random_state=2)

# Adjusting hyperparameters using grid search
grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='recall', verbose=4)
grid_search.fit(X_train, y_train)

# Print the best score and the best parameters
print("Best score:", grid_search.best_score_)
print("Best params:", grid_search.best_params_)

# Training Random Forest Models with Optimal Parameters
rf = RandomForestClassifier(**grid_search.best_params_, random_state=2)
rf.fit(X_train, y_train)

# Predictive Test Sets
y_pred = rf.predict(X_test)

# Output assessment indicators
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("F2 Score:", fbeta_score(y_test, y_pred, beta=2))

