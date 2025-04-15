import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
import joblib  # To save the model and the encoder

# Load the dataset from a CSV file
df = pd.read_csv('path_to_training_csv_file.csv')

# Assuming that the last column is the target (class label), and the rest are features
X = df.iloc[:, :-1]  # Features (all columns except the last)
y = df.iloc[:, -1]   # Target variable (last column)

# Encoding the target variable in case it's categorical
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Save the encoded target variable

# Create an XGBoost classifier
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Initialize Stratified K-Fold Cross-Validation
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation and get accuracy for each fold
cv_scores = cross_val_score(model, X, y_encoded, cv=stratified_kfold, scoring='accuracy')

# Output the cross-validation scores and average accuracy
print(f"\nCross-validation scores: {cv_scores}")
print(f"Mean cross-validation accuracy: {cv_scores.mean() * 100:.2f}%")

# Optionally, you can train the model on the full dataset after evaluating with cross-validation
model.fit(X, y_encoded)

# Save the trained model and the label encoder to .pkl files
model_filename = 'xgboost.pkl'
encoder_filename = 'xgboost_label_encoder.pkl'

# Save both the model and the encoder
joblib.dump(model, model_filename)
joblib.dump(label_encoder, encoder_filename)

print(f"Model saved as {model_filename}")
print(f"LabelEncoder saved as {encoder_filename}")

# Make predictions on the entire dataset (this is typically done for final evaluation)
y_pred = model.predict(X)

print("Training is completed.")
