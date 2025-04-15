import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Function to load models
def load_cnn_model(model_path):
    return load_model(model_path)  # Load cnn model using Keras load_model()

# Load the trained cnn model
cnn_model_path = 'trained models/cnn/cnn_category_train.h5'  # Path to your saved cnn model
cnn_model = load_cnn_model(cnn_model_path)

# Load the scaler and label encoder for category prediction
with open('trained models/cnn/cnn_scaler_category_train.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('trained models/cnn/cnn_label_encoder_category_train.pkl', 'rb') as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Load website prediction models for each category (you should have separate models for each category)
website_models = {}
website_scalers = {}
website_encoders = {}
categories = ['cluster0', 'cluster1', 'cluster2']  # Example categories

# Load each website model, scaler, and label encoder for the corresponding categories
for category in categories:
    # Load website model for each category
    website_model_path = f'trained models/cnn/cnn_{category}.h5'  # Update the path
    website_models[category] = load_model(website_model_path)
    
    # Load scaler for website model for each category
    with open(f'trained models/cnn/cnn_scaler_{category}.pkl', 'rb') as scaler_file:
        website_scalers[category] = pickle.load(scaler_file)
    
    # Load label encoder for website model for each category
    with open(f'trained models/cnn/cnn_label_encoder_{category}.pkl', 'rb') as encoder_file:
        website_encoders[category] = pickle.load(encoder_file)

# Read the test data from CSV file
test_data = pd.read_csv('path_to_test_csv_file.csv')

# Define the feature names (same as in your training process)
feature_names = [
    'Total_number_of_packets', 'Incoming_packets', 'Outgoing_packets',
    'Fraction_of_incoming_packets', 'Fraction_of_outgoing_packets',
    'Total_transmission_size', 'Average_packet_size', 'Total_incoming_size',
    'Total_outgoing_size', 'Avg_outgoing_packet_size', 'Std_dev_outgoing_packet_size',
    'Avg_incoming_packet_size', 'Std_dev_incoming_packet_size', 'Total_transmission_time',
    'Avg_transmission_time', 'Variance_transmission_time', 'Std_dev_transmission_time',
    'Min_transmission_time', 'Max_transmission_time', 'Median_transmission_time',
    'Skewness_transmission_time', 'Rate_of_packet_arrival', 'Median_packet_length',
    'Mean_inter_arrival_time', 'Variance_inter_arrival_time', 'Std_dev_inter_arrival_time',
    'Min_inter_arrival_time', 'Median_inter_arrival_time', 'Skewness_inter_arrival_time',
    'Average_burst_duration', 'Burst_frequency', 'Entropy_of_packet_sizes',
    'Incoming_to_outgoing_packet_ratio', 'Incoming_to_outgoing_size_ratio',
    'Average_bytes_per_packet', 'Min_packet_size', 'Max_packet_size', 'Mode_of_packet_sizes',
    '75th_Percentile_packet_size', '25th_Percentile_packet_size'
]

# Initialize counters for correct predictions
category_correct_count = 0
website_correct_count = 0
denominator = 0

# Create a list to store the results for each row
results = []

# Iterate over each row in the test data
for index, row in test_data.iterrows():
    # Features and target columns
    features = row[:-2]  # All columns except the last two (assumed to be 'Website_category' and 'Website_name')
    true_category = row['Website_category']
    true_website = row['Website_name']

    # Convert the features to a DataFrame with the correct feature names
    features_df = pd.DataFrame([features], columns=feature_names)

    # Step 1: Feature scaling for category prediction
    features_scaled = scaler.transform(features_df)

    # Step 2: Predict the category using the cnn model
    predicted_category_probabilities = cnn_model.predict(features_scaled)
    predicted_category = np.argmax(predicted_category_probabilities, axis=1)[0]  # Get the category with the highest probability

    # Step 3: Convert predicted category back to original label
    predicted_category_label = label_encoder.inverse_transform([predicted_category])[0]

    # Step 4: Predict the website name based on predicted category (use the corresponding website model)
    predicted_website = ""
    
    # Check if we have a model for the predicted category
    if predicted_category_label in website_models:
        # Get the corresponding website model for the predicted category
        website_model = website_models[predicted_category_label]
        
        # Step 5: Feature scaling for website prediction (use the appropriate scaler for this category)
        website_features_scaled = website_scalers[predicted_category_label].transform(features_df)
        
        # Predict the website name using the website model
        predicted_website_probabilities = website_model.predict(website_features_scaled)
        predicted_website_label_encoded = np.argmax(predicted_website_probabilities, axis=1)[0]  # Get the website with highest probability
        
        # Convert the predicted website label back to the original website name using the label encoder
        predicted_website = website_encoders[predicted_category_label].inverse_transform([predicted_website_label_encoded])[0]

    # Step 6: Check if category prediction is correct
    denominator += 1
    if predicted_category_label == true_category:
        category_correct_count += 1

        # Step 7: Check if website prediction is correct
        if predicted_website == true_website:
            website_correct_count += 1

    # Append the results for this row to the results list
    results.append({
        'Index': denominator,
        'True_Category': true_category,
        'Predicted_Category': predicted_category_label,
        'True_Website': true_website,
        'Predicted_Website': predicted_website
    })

    print(f"Index: {denominator}, True Category: {true_category}, Predicted Category: {predicted_category_label}, "
          f"True Website: {true_website}, Predicted Website: {predicted_website}")

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# Save the results to an Excel file
results_df.to_excel('prediction_results_cnn_with_website.xlsx', index=False)

# Print the counts of correctly predicted categories and websites
print("\nCorrect category predictions:", category_correct_count)
print("\nCorrect website predictions:", website_correct_count)

# Calculate accuracy for category and website
category_accuracy = category_correct_count / denominator if denominator > 0 else 0
website_accuracy = website_correct_count / denominator if denominator > 0 else 0

print(f"\nOverall category prediction accuracy: {category_accuracy:.4f}")
print(f"Overall website prediction accuracy: {website_accuracy:.4f}")