import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dropout, GlobalAveragePooling1D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2  # Import L2 Regularization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight  # For class imbalance handling
import pickle

# Load the CSV file
file_path = 'path_to_train_csv_file.csv'
data = pd.read_csv(file_path)

X = data.iloc[:, :-1].values  # Features
y = data['Website_name'].values  # Target column name
# target column name will be Website_category for category prediction model

# Encoding target variable (converting to integer labels for each class)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Converts the class labels to integers

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshaping input for 1D CNN (adding an extra dimension for channels)
X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

# Compute class weights to handle class imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(enumerate(class_weights))

# Define K-Folds cross-validator
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store cross-validation results
val_accuracies = []

# Cross-validation loop
for train_idx, val_idx in kfold.split(X_scaled, y):
    # Split the data into training and validation sets for this fold
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # One-hot encode the labels for multiclass classification (after the split)
    # ------------------------------------------------------------------------------------------------------------------------
    y_train = to_categorical(y_train, num_classes=14)  # Adjust num_classes as the number of classes for prediction
    y_val = to_categorical(y_val, num_classes=14)
    
    # Build the model (reinitialize to avoid reusing previous weights)
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], 1)))
    
    # Adding 1D Convolutional layers with L2 regularization
    model.add(Conv1D(128, kernel_size=5, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))

    model.add(Conv1D(256, kernel_size=5, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))

    # Global Average Pooling instead of flattening
    model.add(GlobalAveragePooling1D())
    model.add(Dense(128, activation='relu'))
    
    
    # ------------------------------------------------------------------------------------------------------------------------
    model.add(Dense(14, activation='softmax'))  # 14 output units for 14 classes
    #adjust number 14 as number of prediction classes

    # Compile the model with the Adam optimizer
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Add callbacks: Early stopping and ReduceLROnPlateau
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    # Train the model on the current fold
    history = model.fit(X_train, y_train, epochs=100, batch_size=64,
                        validation_data=(X_val, y_val),
                        class_weight=class_weight_dict,  # Using class weights
                        callbacks=[early_stopping, reduce_lr], verbose=1)

    # Store validation accuracy for this fold
    val_accuracies.append(history.history['val_accuracy'][-1])

# Print the mean and standard deviation of the validation accuracies across all folds
print(f"Cross-validation mean accuracy: {np.mean(val_accuracies)}")
print(f"Cross-validation standard deviation: {np.std(val_accuracies)}")

# One-hot encode the labels for multiclass classification
# --------------------------------------------------------------------------------------------------------------------------------
y_one_hot = to_categorical(y, num_classes=14)  # Ensure the number of classes is correct
# adjust number_classes as number of prediction classes

# Train the final model on the full dataset
model = Sequential()
model.add(Input(shape=(X_scaled.shape[1], 1)))
model.add(Conv1D(128, kernel_size=5, activation='relu', kernel_regularizer=l2(0.001)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))

model.add(Conv1D(256, kernel_size=5, activation='relu', kernel_regularizer=l2(0.001)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))

model.add(GlobalAveragePooling1D())
model.add(Dense(128, activation='relu'))

# ---------------------------------------------------------------------------------------------------------------------------------------
model.add(Dense(14, activation='softmax'))  # 14 output units for 14 classes
# adjust number_classes as number of prediction classes

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.0001)

# Train on the full dataset
history = model.fit(X_scaled, y_one_hot, epochs=100, batch_size=64,
                    class_weight=class_weight_dict,  # Using class weights
                    callbacks=[early_stopping, reduce_lr], verbose=1)

# Save the model to a .h5 file
model_filename = 'cnn_model.h5'
model.save(model_filename)  # Saves the complete model (architecture + weights)

# Saving additional components (scaler and label encoder) for future use
with open('scaler_cnn.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

with open('label_encoder_cnn.pkl', 'wb') as encoder_file:
    pickle.dump(label_encoder, encoder_file)

print(f"Model and additional components saved successfully.")
