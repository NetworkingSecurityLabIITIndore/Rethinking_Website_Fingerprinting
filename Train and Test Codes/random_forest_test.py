import pandas as pd
import joblib  # Correctly use joblib to load models

# Function to load models
def load_model(file_path):
    model = joblib.load(file_path)  # Use joblib to load the model
    return model

# Load the category prediction model
category_model_path = 'trained models/random_forest/category.pkl'  # Path to your category model
category_model = load_model(category_model_path)

# Load the website prediction models for each category
website_models = {}
categories =  ['cluster0', 'cluster1', 'cluster2']

for category in categories:
    website_model_path = f'trained models/random_forest/{category}.pkl'  # Path for each category's website model
    website_models[category] = load_model(website_model_path)

# Read the test data from CSV file
test_data = pd.read_csv('path_to_test_csv_file.csv')

# Get the feature names from the training data (using the first row as an example)
# Assuming that the training data has the same structure as the test data
feature_names = [
    'Total_number_of_packets',
    'Incoming_packets',
    'Outgoing_packets',
    'Fraction_of_incoming_packets',
    'Fraction_of_outgoing_packets',
    'Total_transmission_size',
    'Average_packet_size',
    'Total_incoming_size',
    'Total_outgoing_size',
    'Avg_outgoing_packet_size',
    'Std_dev_outgoing_packet_size',
    'Avg_incoming_packet_size',
    'Std_dev_incoming_packet_size',
    'Total_transmission_time',
    'Avg_transmission_time',
    'Variance_transmission_time',
    'Std_dev_transmission_time',
    'Min_transmission_time',
    'Max_transmission_time',
    'Median_transmission_time',
    'Skewness_transmission_time',
    'Rate_of_packet_arrival',
    'Median_packet_length',
    'Mean_inter_arrival_time',
    'Variance_inter_arrival_time',
    'Std_dev_inter_arrival_time',
    'Min_inter_arrival_time',
    'Median_inter_arrival_time',
    'Skewness_inter_arrival_time',
    'Average_burst_duration',
    'Burst_frequency',
    'Entropy_of_packet_sizes',
    'Incoming_to_outgoing_packet_ratio',
    'Incoming_to_outgoing_size_ratio',
    'Average_bytes_per_packet',
    'Min_packet_size',  
    'Max_packet_size',  
    'Mode_of_packet_sizes',  
    '75th_Percentile_packet_size', 
    '25th_Percentile_packet_size' 
]

# Convert the list of feature names into a pandas DataFrame
feature_names_df = pd.DataFrame(feature_names, columns=['Feature_Name'])
# Initialize counters for each category and website
category_correct_count = {category: 0 for category in categories}
website_correct_count = {category: {} for category in categories}

# Initialize denominator
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

    # Step 1: Predict the category
    predicted_category = category_model.predict(features_df)[0]  # Get the predicted category
    predicted_website = ""
    
    # Increment denominator for category prediction step
    denominator += 1

    # Step 2: Check if category prediction is correct
    if predicted_category == true_category:
        # Increment the correct count for the category
        category_correct_count[predicted_category] += 1

        # If category is correct, predict the website
        predicted_website = website_models[predicted_category].predict(features_df)[0]
        
        # Check if the website prediction is correct
        if predicted_website == true_website:
            # Increment the correct count for the website in the respective category
            if true_category not in website_correct_count:
                website_correct_count[true_category] = {}
            if true_website not in website_correct_count[true_category]:
                website_correct_count[true_category][true_website] = 0
            website_correct_count[true_category][true_website] += 1

    # Append the results for this row to the results list
    results.append({
        'Index': denominator,
        'True_Category': true_category,
        'Predicted_Category': predicted_category,
        'True_Website': true_website,
        'Predicted_Website': predicted_website
    })
    
    print(f"Index: {denominator}, True Category: {true_category}, Predicted Category: {predicted_category}, "
          f"True Website: {true_website}, Predicted Website: {predicted_website}")

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# Save the results to an Excel file
results_df.to_excel('prediction_results_random_forest.xlsx', index=False)

# Print the counts of correctly predicted categories and websites
print("\nCorrect category predictions:")
for category, count in category_correct_count.items():
    print(f"{category}: {count} correct predictions")

print("\nCorrect website predictions within each category:")
for category, website_dict in website_correct_count.items():
    print(f"{category}:")
    for website, count in website_dict.items():
        print(f"  {website}: {count} correct predictions")

# Calculate accuracy for category and website
category_accuracy = sum(category_correct_count.values()) / denominator if denominator > 0 else 0
website_accuracy = sum(sum(website_dict.values()) for website_dict in website_correct_count.values()) / denominator if denominator > 0 else 0

print(f"\nOverall category prediction accuracy: {category_accuracy:.4f}")
print(f"Overall website prediction accuracy: {website_accuracy:.4f}")