# Sampling Techniques for Imbalanced Credit Card Fraud Detection

This project provides a comprehensive evaluation of multiple sampling strategies for handling imbalanced data in credit card fraud detection. The analysis compares 5 sampling techniques across 8 machine learning models, with detailed performance metrics and visualizations.

## Dataset
- **File**: `Creditcard_data.csv`
- **Target**: `Class` (fraud vs. non‑fraud)
- **Challenge**: Highly imbalanced dataset requiring sampling techniques

## Sampling Techniques
The following resampling methods are implemented and compared:
- **Random Over-Sampling**: Duplicates minority class samples
- **SMOTE** (Synthetic Minority Over-sampling Technique): Creates synthetic minority samples
- **ADASYN** (Adaptive Synthetic Sampling): Focuses on harder-to-learn examples
- **Random Under-Sampling**: Reduces majority class samples
- **NearMiss**: Intelligent under-sampling based on nearest neighbors

## Machine Learning Models
Eight classifiers are evaluated with each sampling technique:
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- XGBoost
- Naive Bayes
- Gradient Boosting

## Evaluation Metrics
Comprehensive performance analysis including:
- **Accuracy**: Overall correctness
- **Precision**: Positive prediction accuracy
- **Recall**: True positive detection rate
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **Training Time**: Computational efficiency

## Project Structure
```
Sampling/
├── Sampling_Assignment.ipynb    # Main Jupyter notebook with complete analysis
├── sampling_assignment.py        # Exported Python script
├── Creditcard_data.csv          # Credit card fraud dataset
├── requirements.txt             # Python dependencies
├── plots/                       # Generated visualizations
│   ├── accuracy_bar.png
│   ├── f1_bar.png
│   ├── sampling_boxplot.png
│   ├── roc_curve.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── avg_sampling_accuracy.png
│   └── training_time.png
└── README.md
```

## Environment Setup
Create a virtual environment and install dependencies:

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Linux/Mac:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Required Packages
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn
- xgboost

## How to Run

### Option 1: Jupyter Notebook (Recommended)
Open and run the notebook for interactive exploration:
```bash
jupyter notebook Sampling_Assignment.ipynb
```

### Option 2: Python Script
Run the exported script:
```bash
python sampling_assignment.py
```

## Analysis Workflow
1. **Data Loading & Exploration**: Load the credit card dataset and visualize class distribution
2. **Data Preprocessing**: Standardize features using StandardScaler
3. **Sampling Application**: Apply 5 different sampling techniques to balance the dataset
4. **Model Training**: Train 8 different models on each sampled dataset
5. **Performance Evaluation**: Calculate accuracy, precision, recall, F1, and ROC-AUC scores
6. **Visualization**: Generate comprehensive plots comparing all techniques
7. **Best Model Selection**: Identify the best model-sampling combination

## Key Features
- ✅ Comprehensive comparison of 5 sampling techniques
- ✅ Evaluation across 8 different ML models
- ✅ Multiple performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- ✅ Detailed visualizations saved to `plots/` folder
- ✅ Class distribution analysis before and after sampling
- ✅ Cross-validation for robust evaluation
- ✅ Feature importance analysis
- ✅ Training time comparison

## Results & Visualizations
All visualizations are automatically saved to the `plots/` folder with high resolution (300 DPI).

### 1. Accuracy Comparison
![Accuracy Comparison](plots/accuracy_bar.png)
*Compares accuracy of all models across different sampling techniques*

### 2. F1 Score Comparison
![F1 Score Comparison](plots/f1_bar.png)
*F1 scores showing the balance between precision and recall*

### 3. Sampling Stability Analysis
![Sampling Stability](plots/sampling_boxplot.png)
*Box plot showing accuracy distribution and stability of each sampling technique*

### 4. ROC Curve
![ROC Curve](plots/roc_curve.png)
*Receiver Operating Characteristic curve for the best model*

### 5. Confusion Matrix
![Confusion Matrix](plots/confusion_matrix.png)
*Confusion matrix for the best performing model-sampling combination*

### 6. Feature Importance
![Feature Importance](plots/feature_importance.png)
*Feature importance analysis using Random Forest*

### 7. Average Accuracy by Sampling Technique
![Average Sampling Accuracy](plots/avg_sampling_accuracy.png)
*Average performance of each sampling technique across all models*

### 8. Training Time Comparison
![Training Time](plots/training_time.png)
*Computational efficiency: average training time per model*

## Key Findings
- **Best Sampling Technique**: Determined based on average performance across all models
- **Best Model**: Identified through comprehensive metric evaluation
- **Stability Analysis**: Variance in accuracy reveals technique reliability
- **Time-Performance Tradeoff**: Balance between accuracy and training time

## Notes
- All plots are saved as high-resolution PNG files (300 DPI) in the `plots/` folder
- The notebook includes additional exploratory data analysis with correlation heatmaps
- Cross-validation is performed for robust model evaluation
- The script automatically creates a `plots.zip` file containing all visualizations for easy download

## Future Enhancements
- Hyperparameter tuning using GridSearchCV
- Cost-sensitive learning approaches
- Ensemble methods combining multiple sampling techniques
- Deep learning models for fraud detection

## License
MIT License. See [LICENSE](LICENSE) for details.

## Author
Arjun Angirus

## References
- Credit Card Fraud Detection Dataset
- imbalanced-learn documentation
- scikit-learn documentation
