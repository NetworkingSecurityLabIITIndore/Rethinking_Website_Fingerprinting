from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib  # To save the model

# Directly using the provided CSV file path
csv_path = 'path_to_training_csv_file.csv'

# Load the data
data = pd.read_csv(csv_path)

# Assuming the last column is the target label and all other columns are features
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target labels

# Check for missing values in the feature dataset
print("Missing values in X:")
print(X.isnull().sum())  # This will show how many missing values are in each column of X

# Fill missing values with a constant value, e.g., 0
X = X.fillna(0)

# Dynamically define a range for 'n_neighbors'
k_range = [1,3,5,7,9,11,13]  # Testing k values from 1 to 20

# Define hyperparameter grid, using dynamic range for k
param_grid = {
    'n_neighbors': k_range,  # The range of k values
    'weights': ['uniform','distance'],  # Use uniform or distance-based weighting
    'p': [1,2]  # p=1 for Manhattan distance, p=2 for Euclidean distance
}

# Create a KNN classifier
knn_classifier = KNeighborsClassifier()

# Create StratifiedKFold cross-validator
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Create GridSearchCV object to find the best parameters using stratified k-fold cross-validation
grid_search = GridSearchCV(estimator=knn_classifier, param_grid=param_grid, cv=stratified_kfold, n_jobs=-1)

# Fit the Grid Search to the complete data (training on the whole dataset)
grid_search.fit(X, y)

# Get the best model from the grid search
best_model = grid_search.best_estimator_

# Save the trained model to a .pkl file
model_filename = 'knn.pkl'
joblib.dump(best_model, model_filename)

print(f"Model saved as {model_filename}")

# Optional: Print the best parameters found by GridSearchCV
print(f"\nBest parameters found by GridSearchCV: {grid_search.best_params_}")

# Optional: You can also print the best score achieved during grid search
print(f"\nBest cross-validation score: {grid_search.best_score_}")