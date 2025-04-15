import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib  # For saving the model

# Directly using the provided CSV file path
csv_path = 'path_to_training_csv_file.csv'

# Load the data
data = pd.read_csv(csv_path)

# Assuming the last column is the target label and all other columns are features
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target

# Check for missing values in the feature dataset
print("Missing values in X:")
print(X.isnull().sum())  # This will show how many missing values are in each column of X

# Fill missing values with a constant value, e.g., 0
X = X.fillna(0)

# Initialize Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  # You can change n_estimators, max_depth, etc.

# Initialize Stratified K-Fold Cross-Validation
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation and get accuracy for each fold
cv_scores = cross_val_score(rf_model, X, y, cv=stratified_kfold, scoring='accuracy')

# Output the cross-validation scores and average accuracy
print(f"\nCross-validation scores: {cv_scores}")
print(f"Mean cross-validation accuracy: {cv_scores.mean() * 100:.2f}%")

# Optionally, you can train the model on the full dataset after evaluating with cross-validation
rf_model.fit(X, y)

# Save the trained model to a .pkl file
model_filename = 'random_forest.pkl'
joblib.dump(rf_model, model_filename)

print(f"\nModel saved as {model_filename}")