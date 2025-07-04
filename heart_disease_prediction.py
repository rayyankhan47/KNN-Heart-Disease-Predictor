"""
Heart Disease Prediction Using K-Nearest Neighbors (KNN)
Author: Rayyan Khan

This script implements a machine learning model to predict heart disease using the KNN algorithm.
It includes data preprocessing, feature selection, model training, and evaluation.

REQUIRED PACKAGES:
Install the following packages before running this script:

pip install pandas numpy matplotlib seaborn scikit-learn

Or install all at once:
pip install -r requirements.txt

If you don't have pip installed, you can install it from: https://pip.pypa.io/en/stable/installation/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore')

class HeartDiseasePredictor:
    """
    A class to handle heart disease prediction using KNN algorithm.
    """
    
    def __init__(self, data_path='heart.csv'):
        """
        Initialize the predictor with the dataset path.
        
        Args:
            data_path (str): Path to the heart disease dataset
        """
        self.data_path = data_path
        self.df = None
        self.df_clean = None
        self.model = None
        self.scaler = None
        self.features = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        """Load and display basic information about the dataset."""
        print("Loading heart disease dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        print("\nFirst few rows:")
        print(self.df.head())
        print("\nDataset info:")
        print(self.df.info())
        return self.df
    
    def explore_data(self):
        """Perform exploratory data analysis."""
        print("\n=== EXPLORATORY DATA ANALYSIS ===")
        
        # Basic statistics
        print("\nDescriptive statistics:")
        print(self.df.describe())
        
        # Check for data quality issues
        print("\nChecking for data quality issues...")
        print(f"RestingBP with 0 values: {len(self.df[self.df['RestingBP']==0])}")
        print(f"Cholesterol with 0 values: {len(self.df[self.df['Cholesterol']==0])}")
        
        # Categorical variables analysis
        categorical_cols = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope', 'HeartDisease']
        
        print("\nCategorical variables distribution:")
        for col in categorical_cols:
            print(f"\n{col}:")
            print(self.df[col].value_counts())
        
        return self.df
    
    def clean_data(self):
        """Clean the dataset by handling missing values and encoding categorical variables."""
        print("\n=== DATA CLEANING ===")
        
        # Create a copy for cleaning
        self.df_clean = self.df.copy()
        
        # Remove rows with RestingBP = 0 (only 1 row)
        self.df_clean = self.df_clean[self.df_clean['RestingBP'] != 0]
        print(f"Removed {len(self.df) - len(self.df_clean)} rows with RestingBP = 0")
        
        # Handle missing cholesterol values (0s) with stratified imputation
        hd_mask = self.df_clean['HeartDisease'] == 0
        
        cholesterol_wo_hd = self.df_clean.loc[hd_mask, 'Cholesterol']
        cholesterol_w_hd = self.df_clean.loc[~hd_mask, 'Cholesterol']
        
        self.df_clean.loc[hd_mask, 'Cholesterol'] = cholesterol_wo_hd.replace(
            to_replace=0, value=cholesterol_wo_hd.median()
        )
        self.df_clean.loc[~hd_mask, 'Cholesterol'] = cholesterol_w_hd.replace(
            to_replace=0, value=cholesterol_w_hd.median()
        )
        
        print("Imputed missing cholesterol values with stratified medians")
        
        # One-hot encode categorical variables
        self.df_clean = pd.get_dummies(self.df_clean, drop_first=True)
        print(f"Encoded categorical variables. New shape: {self.df_clean.shape}")
        
        return self.df_clean
    
    def select_features(self, correlation_threshold=0.30):
        """Select features based on correlation with target variable."""
        print(f"\n=== FEATURE SELECTION (correlation > {correlation_threshold}) ===")
        
        correlations = abs(self.df_clean.corr())
        target_correlations = correlations['HeartDisease'].sort_values(ascending=False)
        
        # Select features with correlation above threshold
        selected_features = target_correlations[target_correlations > correlation_threshold].index.tolist()
        selected_features.remove('HeartDisease')  # Remove target variable
        
        self.features = selected_features
        print(f"Selected features: {self.features}")
        print(f"Feature correlations with HeartDisease:")
        for feature in self.features:
            print(f"  {feature}: {target_correlations[feature]:.3f}")
        
        return self.features
    
    def prepare_data(self, test_size=0.15, random_state=417):
        """Prepare training and testing datasets."""
        print(f"\n=== DATA PREPARATION (test_size={test_size}) ===")
        
        X = self.df_clean.drop(['HeartDisease'], axis=1)
        y = self.df_clean['HeartDisease']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.y_test)}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self):
        """Train the KNN model with hyperparameter optimization."""
        print("\n=== MODEL TRAINING ===")
        
        # Scale the features
        self.scaler = MinMaxScaler()
        X_train_scaled = self.scaler.fit_transform(self.X_train[self.features])
        
        # Define hyperparameter grid
        grid_params = {
            'n_neighbors': range(1, 20),
            'metric': ['minkowski', 'manhattan']
        }
        
        # Perform grid search
        knn = KNeighborsClassifier()
        knn_grid = GridSearchCV(knn, grid_params, scoring='accuracy', cv=5)
        knn_grid.fit(X_train_scaled, self.y_train)
        
        self.model = knn_grid.best_estimator_
        
        print(f"Best parameters: {knn_grid.best_params_}")
        print(f"Best cross-validation score: {knn_grid.best_score_*100:.2f}%")
        
        return self.model
    
    def evaluate_model(self):
        """Evaluate the model on test data."""
        print("\n=== MODEL EVALUATION ===")
        
        # Scale test data
        X_test_scaled = self.scaler.transform(self.X_test[self.features])
        
        # Make predictions
        predictions = self.model.predict(X_test_scaled)
        
        # Calculate accuracy
        accuracy = accuracy_score(self.y_test, predictions)
        print(f"Test Accuracy: {accuracy*100:.2f}%")
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, predictions)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # Calculate additional metrics
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print(f"\nSensitivity (True Positive Rate): {sensitivity*100:.2f}%")
        print(f"Specificity (True Negative Rate): {specificity*100:.2f}%")
        
        return accuracy, cm
    
    def plot_confusion_matrix(self):
        """Plot the confusion matrix."""
        X_test_scaled = self.scaler.transform(self.X_test[self.features])
        predictions = self.model.predict(X_test_scaled)
        cm = confusion_matrix(self.y_test, predictions)
        
        plt.figure(figsize=(8, 6))
        ConfusionMatrixDisplay(cm).plot()
        plt.title('Confusion Matrix - Heart Disease Prediction')
        plt.show()
    
    def plot_correlation_heatmap(self):
        """Plot correlation heatmap for selected features."""
        correlations = abs(self.df_clean[self.features + ['HeartDisease']].corr())
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlations, annot=True, cmap='rocket_r', center=0.5)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.show()
    
    def run_complete_analysis(self, plot_results=True):
        """Run the complete analysis pipeline."""
        print("=== HEART DISEASE PREDICTION ANALYSIS ===\n")
        
        # Load and explore data
        self.load_data()
        self.explore_data()
        
        # Clean and prepare data
        self.clean_data()
        self.select_features()
        self.prepare_data()
        
        # Train and evaluate model
        self.train_model()
        accuracy, cm = self.evaluate_model()
        
        # Plot results if requested
        if plot_results:
            self.plot_correlation_heatmap()
            self.plot_confusion_matrix()
        
        print(f"\n=== ANALYSIS COMPLETE ===")
        print(f"Final Model Accuracy: {accuracy*100:.2f}%")
        
        return accuracy, cm

def main():
    """Main function to run the heart disease prediction analysis."""
    # Create predictor instance
    predictor = HeartDiseasePredictor()
    
    # Run complete analysis
    accuracy, confusion_matrix = predictor.run_complete_analysis(plot_results=True)
    
    print(f"\nAnalysis completed successfully!")
    print(f"Model achieved {accuracy*100:.2f}% accuracy on test data.")

if __name__ == "__main__":
    main() 