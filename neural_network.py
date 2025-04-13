# Task 2: Neural Network Classification of Diabetes
# This script implements a full solution including data preparation, neural network implementation,
# model evaluation, feature importance analysis, and hyperparameter tuning

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
# Updated import for newer TensorFlow versions
from scikeras.wrappers import KerasClassifier
import shap

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# 1. Data Preparation
print("Step 1: Data Preparation")

# Load the dataset (adjust path if needed)
try:
    data = pd.read_csv('diabetes.csv')
except FileNotFoundError:
    print("Dataset not found. Please ensure 'diabetes.csv' is in the current directory.")
    exit()

print("Dataset shape:", data.shape)
print("First 5 rows:")
print(data.head())

# Check for missing values
print("\nMissing values:")
print(data.isnull().sum())

# Check zero values that might represent missing data
print("\nZero values that might represent missing data:")
for column in data.columns:
    if column != 'Outcome':  # Skip the target variable
        zero_count = (data[column] == 0).sum()
        if zero_count > 0:
            print(f"{column}: {zero_count} zeros")

# Handle missing values (zeros in some columns are physiologically impossible)
# Replace zeros with NaN for these columns and then fill with median
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for column in cols_with_zeros:
    # Replace zeros with NaN
    data[column] = data[column].replace(0, np.nan)
    # Fill NaN with median
    data[column] = data[column].fillna(data[column].median())

print("\nAfter handling missing values, zero counts:")
for column in cols_with_zeros:
    print(f"{column}: {(data[column] == 0).sum()} zeros")

# Separate features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Apply feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into train, validation, and test sets
# First split into train+val and test (80% / 20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Then split train+val into train and validation (87.5% / 12.5% of the 80%, which is 70% / 10% of the total)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.125, random_state=42)

print(f"\nData split:\n- Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X_scaled):.1%})")
print(f"- Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X_scaled):.1%})")
print(f"- Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X_scaled):.1%})")

# 2. Neural Network Implementation
print("\nStep 2: Neural Network Implementation")

def create_model(neurons=16, learning_rate=0.001, optimizer='adam'):
    """Create a feedforward neural network model for binary classification."""
    model = Sequential([
        Dense(neurons, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(neurons // 2, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    # Set optimizer
    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    
    # Compile the model
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Create the initial model
model = create_model()
print(model.summary())

# Implement early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Train the model with batch training
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=2
)

# Plot training history
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.close()

# 3. Model Evaluation
print("\nStep 3: Model Evaluation")

# Make predictions on the test set
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Calculate precision, recall, and F1-score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# Generate detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

# 4. Feature Importance Analysis
print("\nStep 4: Feature Importance Analysis")

# 4.1 SHAP (Shapley Additive Explanations)
# Create a background dataset for SHAP
try:
    # Try using DeepExplainer (works better with TensorFlow models)
    # Select a subset of training data for background
    background_data = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
    
    explainer = shap.DeepExplainer(model, background_data)
    # Take a subset of test data for efficiency
    test_sample = X_test[:100]
    shap_values = explainer.shap_values(test_sample)
    
    # For DeepExplainer with Keras models, shap_values is usually a list with one element
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    # Create SHAP summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, test_sample, feature_names=X.columns, show=False)
    plt.title('SHAP Feature Importance')
    plt.tight_layout()
    plt.savefig('shap_importance.png')
    plt.close()
    
    # Calculate mean absolute SHAP values for each feature
    shap_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': np.abs(shap_values).mean(axis=0)
    })
    shap_importance = shap_importance.sort_values('Importance', ascending=False)
    
except Exception as e:
    print(f"Error in SHAP calculation: {e}")
    print("Falling back to alternative feature importance calculation")
    
    # Alternative approach if SHAP fails
    feature_importance = {}
    for i, feature in enumerate(X.columns):
        # Make a copy of the test data
        X_test_copy = X_test.copy()
        # Permute the feature
        np.random.shuffle(X_test_copy[:, i])
        # Calculate predictions with original and permuted data
        original_pred = model.predict(X_test).flatten()
        permuted_pred = model.predict(X_test_copy).flatten()
        # Mean absolute difference in predictions
        importance = np.mean(np.abs(original_pred - permuted_pred))
        feature_importance[feature] = importance
    
    # Convert to DataFrame and sort
    shap_importance = pd.DataFrame(list(feature_importance.items()), columns=['Feature', 'Importance'])
    shap_importance = shap_importance.sort_values('Importance', ascending=False)

# 4.2 Permutation Importance
print("\nCalculating Permutation Importance...")

# We need to define a custom scoring function for the TensorFlow model
def keras_f1_score(model, X, y):
    """Calculate F1 score for a Keras model"""
    y_pred = (model.predict(X) > 0.5).astype(int).flatten()
    return f1_score(y, y_pred)

# Manual implementation of permutation importance
def manual_permutation_importance(model, X, y, n_repeats=5, random_state=42):
    """Calculate permutation importance for any model"""
    # Set random seed
    np.random.seed(random_state)
    
    # Get original score
    baseline_score = keras_f1_score(model, X, y)
    
    # Initialize importance scores
    n_features = X.shape[1]
    importances = np.zeros(n_features)
    importances_std = np.zeros(n_features)
    
    # For each feature
    for i in range(n_features):
        feature_scores = []
        
        # Repeat n times
        for _ in range(n_repeats):
            # Create a permuted copy
            X_permuted = X.copy()
            # Permute the feature
            X_permuted[:, i] = X_permuted[np.random.permutation(X.shape[0]), i]
            
            # Calculate permuted score
            permuted_score = keras_f1_score(model, X_permuted, y)
            
            # Calculate importance as decrease in performance
            feature_scores.append(baseline_score - permuted_score)
        
        # Calculate mean and std of importance
        importances[i] = np.mean(feature_scores)
        importances_std[i] = np.std(feature_scores)
    
    return importances, importances_std

# Calculate permutation importance
perm_importances, perm_std = manual_permutation_importance(model, X_test, y_test)

# Create dataframe
perm_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': perm_importances,
    'StdDev': perm_std
})
perm_importance_df = perm_importance_df.sort_values('Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(12, 10))

plt.subplot(2, 1, 1)
sns.barplot(x='Importance', y='Feature', data=shap_importance)
plt.title('SHAP Feature Importance')
plt.xlabel('Mean |SHAP Value|')

plt.subplot(2, 1, 2)
sns.barplot(x='Importance', y='Feature', data=perm_importance_df)
plt.title('Permutation Feature Importance')
plt.xlabel('Mean Decrease in F1 Score')

plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

print("Top features by SHAP importance:")
print(shap_importance.head())

print("\nTop features by Permutation importance:")
print(perm_importance_df.head())

# 5. Hyperparameter Tuning
print("\nStep 5: Hyperparameter Tuning")

# Define the model creation function for KerasClassifier
def create_model_for_grid(neurons=16, learning_rate=0.001, optimizer='adam'):
    return create_model(neurons, learning_rate, optimizer)

# Create a KerasClassifier (using scikeras instead of keras.wrappers)
model_for_grid = KerasClassifier(
    model=create_model_for_grid,
    epochs=50,
    batch_size=32,
    verbose=0
)

# Define the hyperparameter grid
param_grid = {
    'model__neurons': [8, 16, 32],
    'model__learning_rate': [0.001, 0.01],
    'model__optimizer': ['adam', 'rmsprop', 'sgd'],
    'batch_size': [16, 32, 64]
}

# Create the grid search
grid = GridSearchCV(
    estimator=model_for_grid,
    param_grid=param_grid,
    cv=3,
    verbose=1,
    n_jobs=1  # Using 1 job for compatibility
)

# Fit the grid search
print("Starting Grid Search for Hyperparameter Tuning (this may take some time)...")
grid_result = grid.fit(X_train, y_train)

# Print results
print(f"Best parameters: {grid_result.best_params_}")
print(f"Best cross-validation score: {grid_result.best_score_:.4f}")

# Extract best parameters (format is different in scikeras)
best_neurons = grid_result.best_params_['model__neurons']
best_learning_rate = grid_result.best_params_['model__learning_rate'] 
best_optimizer = grid_result.best_params_['model__optimizer']
best_batch_size = grid_result.best_params_['batch_size']

print(f"Best hyperparameters: neurons={best_neurons}, learning_rate={best_learning_rate}, "
      f"optimizer={best_optimizer}, batch_size={best_batch_size}")

# Train a final model with the best parameters
final_model = create_model(
    neurons=best_neurons,
    learning_rate=best_learning_rate,
    optimizer=best_optimizer
)

# Train the final model with early stopping
final_history = final_model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=best_batch_size,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=2
)

# Evaluate the tuned model
print("\nFinal Model Evaluation after Hyperparameter Tuning:")
y_pred_final_prob = final_model.predict(X_test)
y_pred_final = (y_pred_final_prob > 0.5).astype(int).flatten()

# Compute final confusion matrix and metrics
cm_final = confusion_matrix(y_test, y_pred_final)
print("Confusion Matrix (Tuned Model):")
print(cm_final)

# Calculate precision, recall, and F1-score for final model
precision_final = precision_score(y_test, y_pred_final)
recall_final = recall_score(y_test, y_pred_final)
f1_final = f1_score(y_test, y_pred_final)

print(f"Precision: {precision_final:.4f}")
print(f"Recall: {recall_final:.4f}")
print(f"F1-score: {f1_final:.4f}")

# Plot final confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_final, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Tuned Model)')
plt.tight_layout()
plt.savefig('confusion_matrix_tuned.png')
plt.close()

# Compare initial and tuned models
print("\nModel Performance Comparison:")
print(f"Initial model - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
print(f"Tuned model - Precision: {precision_final:.4f}, Recall: {recall_final:.4f}, F1: {f1_final:.4f}")

print("\nTask 2 completed successfully!")