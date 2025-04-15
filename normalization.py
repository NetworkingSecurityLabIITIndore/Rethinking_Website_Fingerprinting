import joblib
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os

# Path to your CSV file
csv_file_path = 'Enter_path_to_csv_file.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Want to normalize all columns except the last one ('Website_name')
columns_to_normalize = df.columns[:-1]  

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit the scaler and transform the data
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# Save the scaler to a file (for future use with test data)
scaler_file_path = os.path.join(os.getcwd(), "minmax_scaler.pkl")  # Current directory
joblib.dump(scaler, scaler_file_path)

# Optionally, save the normalized DataFrame to a new CSV file
normalized_csv_file_path = os.path.join(os.getcwd(), "normalized_feature.csv")
df.to_csv(normalized_csv_file_path, index=False)

print("Normalization is done and saved in a normalized_feature.csv")

# Now you have the scaler saved and the normalized data in a new CSV file