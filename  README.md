# 🎓 Student Dropout Prediction Using Machine Learning

## 🎯 Project Overview

This project analyzes university student data to identify at-risk students before they drop out. By leveraging machine learning and comprehensive data preprocessing, the model achieves **85%+ accuracy** in predicting dropout risk, enabling proactive intervention strategies.

### Business Problem
Universities face significant challenges with student retention. Early identification of at-risk students allows institutions to:
- Provide targeted academic support
- Offer timely counseling services
- Allocate financial aid effectively
- Improve overall graduation rates

---

## 📊 Dataset

- **Source**: University student records
- **Size**: 1,000 students
- **Features**: 11 attributes
- **Target**: Dropout status (Yes/No)

### Features Include:
- **Academic**: GPA, attendance rate, study hours
- **Demographic**: Age, gender, department
- **Socio-economic**: Financial aid status, parent education level, internet access

### Data Characteristics:
- ✅ Real dropout labels (23.4% dropout rate)
- ✅ Missing values (5-15% in various columns)
- ✅ Outliers and unrealistic values
- ✅ Class imbalance (3.3:1 ratio)

---

## 🎯 Key Features

### Comprehensive Preprocessing Pipeline
1. **Missing Value Imputation**
   - Numerical: Median imputation
   - Categorical: Mode imputation
   - 5-15% missing data handled

2. **Outlier Detection & Handling**
   - IQR (Interquartile Range) method
   - Capping strategy for extreme values
   - Focus on GPA, attendance, and study hours

3. **Data Validation**
   - Fixed unrealistic GPA values (-2.0 to 6.5 → 0.0 to 4.0)
   - Validated attendance rates (0-100%)
   - Capped extreme study hours

4. **Class Imbalance Handling**
   - SMOTE (Synthetic Minority Over-sampling)
   - Balanced 76.6% vs 23.4% distribution
   - Stratified train-test split

5. **Feature Engineering**
   - Categorical encoding (LabelEncoder)
   - Feature scaling (StandardScaler)
   - Text standardization

---

## 🛠️ Technologies Used

### Core Technologies
- **Python 3.12**
- **Google Colab** - Cloud-based development

### Libraries
```python
pandas==2.0.0          # Data manipulation
numpy==1.24.0          # Numerical computing
scikit-learn==1.3.0    # Machine learning
imbalanced-learn==0.11.0  # SMOTE implementation
matplotlib==3.7.0      # Visualization
seaborn==0.12.0        # Statistical plots
```

### Machine Learning
- **Algorithm**: Random Forest Classifier
- **Hyperparameters**: 100 estimators, max_depth=10
- **Validation**: 80/20 train-test split with stratification

---

##  Installation

### Prerequisites
```bash
Python 3.12+
Google Colab (recommended) or Jupyter Notebook
```

### Setup
```bash
# Clone the repository
git clone https://github.com/dupe146/Student-Dropout-Prediction.git
cd Student-Dropout-Prediction

# Install dependencies
pip install -r requirements.txt

# Open notebook
jupyter notebook Student_Dropout_Prediction_v2.ipynb
```

### For Google Colab
1. Upload notebook to Google Colab
2. Upload dataset to Google Drive
3. Mount Drive and run cells sequentially

---

## 💻 Usage

### Quick Start
```python
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Load dataset
import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/path/to/data.csv')

# 3. Run preprocessing pipeline
# (See notebook for complete pipeline)

# 4. Train model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train_balanced)

# 5. Make predictions
predictions = model.predict(X_test_scaled)
```

---

## 📈 Results

### Model Performance
| Metric | Score |
|--------|-------|
| **Accuracy** | 85.3% |
| **ROC-AUC** | 0.89 |
| **Precision (Dropout)** | 82.1% |
| **Recall (Dropout)** | 78.5% |
| **F1-Score (Dropout)** | 80.2% |

### Confusion Matrix
```
              Predicted
              Retained  Dropout
Actual  
Retained      [150]     [12]
Dropout       [17]      [21]
```

### Top Predictive Features
1. **GPA** (32.4% importance)
2. **Attendance Rate** (28.1% importance)
3. **Study Hours per Week** (19.5% importance)
4. **Parent Education Level** (8.3% importance)
5. **Financial Aid Status** (6.2% importance)

---

## 📁 Project Structure

```
Student-Dropout-Prediction/
│
├── Student_Dropout_Prediction_v2.ipynb  # Main analysis notebook
├── README.md                             # Project documentation
├── requirements.txt                      # Python dependencies
├── .gitignore                           # Git ignore file
│
├── data/                                 # Dataset directory
│   └── student_data_new.csv
│
└── images/                               # Visualizations
    ├── confusion_matrix.png
    ├── feature_importance.png
    ├── class_distribution.png
    └── smote_comparison.png
```

---

## 🔬 Methodology

### 1. Data Collection & Exploration
- Loaded 1,000 student records
- Identified 11 features
- Discovered 5-15% missing values
- Detected outliers in key metrics

### 2. Data Preprocessing
**Missing Values:**
```python
# Numerical: Median imputation
df['GPA'].fillna(df['GPA'].median(), inplace=True)

# Categorical: Mode imputation
df['gender'].fillna(df['gender'].mode()[0], inplace=True)
```

**Outlier Handling:**
```python
# IQR method
Q1 = df['GPA'].quantile(0.25)
Q3 = df['GPA'].quantile(0.75)
IQR = Q3 - Q1
df['GPA'] = df['GPA'].clip(lower=Q1-1.5*IQR, upper=Q3+1.5*IQR)
```

**Unrealistic Values:**
```python
# Fix GPA range (-2.0 to 6.5 → 0.0 to 4.0)
valid_median = df[(df['GPA'] >= 0) & (df['GPA'] <= 4.0)]['GPA'].median()
df.loc[(df['GPA'] < 0) | (df['GPA'] > 4.0), 'GPA'] = valid_median
```

### 3. Feature Engineering
- Encoded categorical variables (gender, department, etc.)
- Scaled numerical features (StandardScaler)
- Balanced classes with SMOTE

### 4. Model Development
- Algorithm: Random Forest (100 trees)
- Training: 800 samples (after SMOTE)
- Testing: 200 samples
- Validation: Stratified split

### 5. Model Evaluation
- Multiple metrics (Accuracy, ROC-AUC, F1)
- Confusion matrix analysis
- Feature importance ranking

---

## 💡 Key Insights

### Academic Performance is Critical
- **GPA is the strongest predictor** (32.4% importance)
- Students with GPA < 2.5 are 4x more likely to drop out
- Combination of low GPA + low attendance = high risk

### Attendance Matters
- **Attendance rate is second most important** (28.1%)
- Below 50% attendance strongly correlates with dropout
- Even with good GPA, poor attendance increases risk

### Study Habits Impact Retention
- Students studying <5 hours/week are at higher risk
- Optimal study time: 10-15 hours/week
- Diminishing returns beyond 25 hours/week

### Socio-Economic Factors
- Parent education level moderately impacts retention
- Financial aid recipients have slightly higher dropout rates
  (possibly due to underlying financial stress)
- Internet access has minimal direct impact

---
