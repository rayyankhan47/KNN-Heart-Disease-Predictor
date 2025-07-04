# Heart Disease Prediction Using K-Nearest Neighbors (KNN)

## Overview

This project implements a machine learning model to predict heart disease using the K-Nearest Neighbors (KNN) algorithm. The model analyzes various clinical parameters to classify patients as having or not having heart disease, achieving an accuracy of **82.61%** on the test set.

## Project Structure

```
heart-disease-knn/
├── README.md                          # This file
├── predict.ipynb                      # Jupyter notebook with complete analysis
├── heart_disease_prediction.py        # Python script version (production-ready)
├── requirements.txt                   # Python dependencies
└── heart.csv                          # Dataset containing patient information
```

## Dataset

The project uses a heart disease dataset with **918 patients** and **12 features**:

### Features
- **Age**: Patient's age in years
- **Sex**: Gender (M/F)
- **ChestPainType**: Type of chest pain (ATA, NAP, ASY, TA)
- **RestingBP**: Resting blood pressure (mm Hg)
- **Cholesterol**: Serum cholesterol (mm/dl)
- **FastingBS**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- **RestingECG**: Resting electrocardiogram results (Normal, ST, LVH)
- **MaxHR**: Maximum heart rate achieved
- **ExerciseAngina**: Exercise-induced angina (Y/N)
- **Oldpeak**: ST depression induced by exercise relative to rest
- **ST_Slope**: Slope of peak exercise ST segment (Up, Flat, Down)
- **HeartDisease**: Target variable (1 = heart disease, 0 = no heart disease)

## Methodology

### 1. Data Exploration and Cleaning
- **Data Quality Issues**: Identified and addressed missing values (0s) in RestingBP and Cholesterol
- **Feature Engineering**: Converted categorical variables to numerical using one-hot encoding
- **Data Imputation**: Replaced missing cholesterol values with median values stratified by heart disease status

### 2. Feature Selection
- **Correlation Analysis**: Used heatmap visualization to identify features with correlation > 0.30
- **Selected Features**:
  - MaxHR (Maximum heart rate)
  - Oldpeak (ST depression)
  - ExerciseAngina_Y (Exercise-induced angina)
  - ST_Slope_Flat (Flat ST slope)
  - ST_Slope_Up (Upward ST slope)

### 3. Model Development
- **Algorithm**: K-Nearest Neighbors Classifier
- **Data Preprocessing**: Min-Max scaling for feature normalization
- **Hyperparameter Tuning**: Grid search for optimal k-value and distance metric
- **Best Parameters**:
  - n_neighbors: 19
  - metric: manhattan

### 4. Model Evaluation
- **Train/Test Split**: 85% training, 15% testing
- **Performance Metrics**:
  - Training Accuracy: 82.53%
  - Test Accuracy: 82.61%
  - Confusion Matrix analysis for healthcare implications

## Results

### Model Performance
- **Overall Accuracy**: 82.61%
- **No Overfitting**: Training and test accuracies are very similar
- **Healthcare Impact**: The model correctly identifies most heart disease cases, with only 11 false negatives (patients predicted as healthy who actually have heart disease)

### Key Insights
1. **ST_Slope_Up** was the most predictive single feature (84.06% accuracy)
2. **Exercise-induced angina** and **ST depression** are strong indicators of heart disease
3. **Maximum heart rate** shows moderate predictive power
4. The model performs well without overfitting, indicating good generalization

## Usage

### Prerequisites

#### Option 1: Install all dependencies at once
```bash
pip install -r requirements.txt
```

#### Option 2: Install packages individually
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

#### Option 3: For Jupyter Notebook only
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

#### Option 4: For Python script only
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Running the Analysis

#### Option 1: Jupyter Notebook (Interactive)
1. Clone this repository
2. Open `predict.ipynb` in Jupyter Notebook or JupyterLab
3. Run all cells to reproduce the analysis with visualizations

#### Option 2: Python Script (Command Line)
```bash
python heart_disease_prediction.py
```

#### Option 3: Import as Module
```python
from heart_disease_prediction import HeartDiseasePredictor

# Create predictor instance
predictor = HeartDiseasePredictor('heart.csv')

# Run complete analysis
accuracy, cm = predictor.run_complete_analysis()

# Or run individual steps
predictor.load_data()
predictor.clean_data()
predictor.train_model()
accuracy = predictor.evaluate_model()
```

### Key Code Sections
- **Data Loading**: `df = pd.read_csv('heart.csv')`
- **Data Cleaning**: Handles missing values and categorical encoding
- **Feature Selection**: Correlation-based feature selection
- **Model Training**: KNN with hyperparameter optimization
- **Evaluation**: Accuracy metrics and confusion matrix

## Technical Details

### Data Preprocessing
- **Missing Value Handling**: Stratified imputation for cholesterol values
- **Feature Scaling**: Min-Max normalization
- **Categorical Encoding**: One-hot encoding for categorical variables

### Model Configuration
- **Algorithm**: KNeighborsClassifier from scikit-learn
- **Distance Metric**: Manhattan distance (L1 norm)
- **Neighbors**: 19 nearest neighbors
- **Cross-validation**: Grid search with accuracy scoring

## Healthcare Implications

This model can assist healthcare professionals in:
- **Early Detection**: Identifying patients at risk of heart disease
- **Clinical Decision Support**: Providing additional data points for diagnosis
- **Resource Allocation**: Prioritizing patients for further cardiac evaluation

**Note**: This model should be used as a supplementary tool, not as a replacement for professional medical diagnosis.

## Author

**Rayyan Khan**

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.