# Neural-Network-Classification-of-Diabetes

This project implements a Neural Network (Multilayer Perceptron, MLP) for binary classification of diabetes using the Pima Indians Diabetes Dataset from Kaggle. The model undergoes preprocessing, training, evaluation, hyperparameter tuning, and feature importance analysis.

🚀 Objectives
Data Preprocessing: Handle missing values, scale features, and split dataset.

Neural Network Model: Train an MLP for diabetes classification.

Model Evaluation: Compute metrics like Precision, Recall, F1-score, and confusion matrix.

Feature Importance Analysis: Use SHAP and Permutation Importance.

Hyperparameter Tuning: Optimize hyperparameters via Grid Search.

🧩 Project Structure
main.py: Main script containing code for:

Data preprocessing

Neural network model creation and training

Model evaluation

Feature importance analysis

Hyperparameter tuning

requirements.txt: List of required Python packages.

🛠️ Requirements
Install dependencies using the following:
pip install -r requirements.txt

📂 Dataset Preparation
Download the Pima Indians Diabetes Dataset from Kaggle.

Load the dataset using pandas.

Handle missing values and scale the features.

Split the data into:

Training set: 70%

Validation set: 10%

Test set: 20%

⚙️ Neural Network Implementation
Feedforward Neural Network (MLP) for binary classification.

Activation functions: ReLU for hidden layers, sigmoid for output.

Loss function: Binary Cross-Entropy.

Optimizer: Adam optimizer.

Training: Use batch training with early stopping to avoid overfitting.

📊 Model Evaluation
Evaluate the model on the test set.

Compute the following metrics:

Precision

Recall

F1-score

Generate a confusion matrix plot.

🔍 Feature Importance Analysis
Techniques:
SHAP (Shapley Additive Explanations) for model explanation.

Permutation Importance to assess the effect of feature permutations on model performance.

Visualize feature importance using a bar plot.

⚙️ Hyperparameter Tuning
Perform hyperparameter optimization for:

Number of neurons

Learning rate

Batch size

Optimizer

Use Grid Search to find the best combination and retrain the model.

📈 Results
Print performance metrics in the console.

Display confusion matrix plot.

Feature importance visualization.

✅ Usage
Run the project using:

python neural_network.py
