# 🎓 Student Dropout Prediction Using Machine Learning

##  Project Overview

This project analyzes university student data to identify at-risk students before they drop out.It demonstrates comprehensive data preprocessing techniques and machine learning model development on real-world messy data.

### **Business Problem**
Universities struggle with student retention. Early identification of at-risk students allows institutions to:
- Provide targeted academic support
- Offer timely counseling services
- Allocate resources effectively
- Improve overall graduation rates

### **Technical Approach**
Binary classification using Random Forest to predict dropout risk based on academic and demographic features.

---

## 📊 Dataset

- **Source**: University student records
- **Size**: 1,000 students
- **Features**: 11 attributes
- **Target**: Dropout status (Yes: 23.4%, No: 76.6%)

### **Features Include:**
- **Academic**: GPA, attendance rate, study hours per week
- **Demographic**: Age, gender, department
- **Socio-economic**: Financial aid status, parent education level, internet access

### **Data Challenges:**
- ✅ Missing values (5-15% across features)
- ✅ Unrealistic values (GPA ranging from -2.0 to 6.5)
- ✅ Class imbalance (3.3:1 ratio)
- ✅ Outliers in multiple features

---

##  Key Features

### **1. Comprehensive Data Preprocessing**

**Missing Value Handling:**
- Numerical features: Median imputation (robust to outliers)
- Categorical features: Mode imputation (preserves common patterns)
- Successfully handled 5-15% missing data across columns

**Unrealistic Value Correction:**
- Identified invalid GPA range (-2.0 to 6.5)
- Established valid range (0.0 to 4.0)
- Replaced 12 invalid entries with median of valid values
- Validated final range consistency

**Outlier Detection & Treatment:**
- Applied IQR (Interquartile Range) method
- Detected outliers in GPA, attendance rate, and study hours
- Used capping strategy (Winsorization) to limit extremes
- Preserved data volume while reducing extreme influence

### **2. Class Imbalance Handling**

**Challenge**: 76.6% retained vs 23.4% dropout

**Solution**: SMOTE (Synthetic Minority Over-sampling Technique)
- Created synthetic examples of minority class
- Balanced training data from 3.3:1 to 1:1 ratio
- Improved model's ability to learn dropout patterns

### **3. Feature Engineering**

**Encoding:**
- Label encoding for categorical variables (gender, department, financial aid, parent education, internet access)
- Binary encoding for target variable (Yes=1, No=0)

**Scaling:**
- StandardScaler normalization (mean=0, std=1)
- Applied to numerical features: GPA, attendance rate, study hours, age
- Prevented scale-based feature dominance

### **4. Model Development & Optimization**

**Initial Approach:**
- Random Forest with default parameters
- Result: Severe overfitting (95% train, 64% test accuracy)

**Optimization:**
- Reduced max_depth (10 → 5)
- Increased min_samples_split (2 → 20)
- Added min_samples_leaf constraint (10)
- Result: Reduced overfitting gap to 17%

---

## 🛠️ Technologies Used

### **Core Technologies**
- **Python 3.12**
- **Google Colab** - Cloud-based development environment

### **Libraries**
```python
pandas==2.0.0              # Data manipulation
numpy==1.24.0              # Numerical computing
scikit-learn==1.3.0        # Machine learning algorithms
imbalanced-learn==0.11.0   # SMOTE implementation
matplotlib==3.7.0          # Visualization
seaborn==0.12.0            # Statistical plots
```

### **Machine Learning**
- **Algorithm**: Random Forest Classifier
- **Configuration**: 100 estimators, max_depth=5, regularization constraints
- **Validation**: 80/20 train-test split with stratification

---

##  Installation

### **Prerequisites**
```bash
Python 3.12+
Google Colab (recommended) or Jupyter Notebook
```

### **Setup**
```bash
# Clone the repository
git clone https://github.com/dupe146/Student-Dropout-Prediction.git
cd Student-Dropout-Prediction

# Install dependencies
pip install -r requirements.txt
```

### **For Google Colab**
1. Upload notebook to Google Colab
2. Mount Google Drive and upload dataset
3. Update data path in Cell 3
4. Run all cells sequentially

---

## 📈 Results

### **Model Performance**

| Metric | Score |
|--------|-------|
| **Test Accuracy** | 58-64% |
| **ROC-AUC** | 0.49-0.50 |
| **Training Accuracy** | 75% (after regularization) |
| **Overfitting Gap** | 17% (significantly improved from initial 31%) |

### **Confusion Matrix**
```
              Predicted
              Retained  Dropout
Actual  
Retained      [117]     [36]
Dropout       [36]      [11]
```

**Interpretation:**
- True Negatives: 117 (correctly predicted retained)
- False Positives: 36 (incorrectly predicted dropout)
- False Negatives: 36 (missed actual dropouts)
- True Positives: 11 (correctly predicted dropout)

### **Feature Importance Rankings**

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Attendance Rate | 18.9% |
| 2 | GPA | 16.8% |
| 3 | Hours Studied per Week | 16.7% |
| 4 | Age | 12.4% |
| 5 | Department | 11.8% |


---

---

## 📁 Project Structure

```
Student-Dropout-Prediction/
│
├── Student_Dropout_Prediction_v2.ipynb  # Main analysis notebook
├── README.md                              # Project documentation
├── requirements.txt                       # Python dependencies
├── .gitignore                            # Git ignore file
│
├── data/                                  # Dataset directory
│   └── student_data_new.csv
│
└── images/                                # Visualizations
    ├── confusion_matrix.png
    ├── feature_importance.png
    └── smote_comparison.png
```

---

## 🔬 Methodology

### **1. Data Loading & Exploration**
- Loaded 1,000 student records with 11 features
- Identified data quality issues (missing values, unrealistic entries)
- Analyzed class distribution (23.4% dropout)

### **2. Preprocessing Pipeline**

**Step 1: Missing Values**
```python
# Numerical: Median imputation
df['GPA'].fillna(df['GPA'].median(), inplace=True)

# Categorical: Mode imputation
df['parent_education_level'].fillna(df['parent_education_level'].mode()[0], inplace=True)
```

**Step 2: Unrealistic Values**
```python
# Fix GPA range
valid_median = df[(df['GPA'] >= 0) & (df['GPA'] <= 4.0)]['GPA'].median()
df.loc[(df['GPA'] < 0) | (df['GPA'] > 4.0), 'GPA'] = valid_median
```

**Step 3: Outliers**
```python
# IQR method with capping
Q1 = df['GPA'].quantile(0.25)
Q3 = df['GPA'].quantile(0.75)
IQR = Q3 - Q1
df['GPA'] = df['GPA'].clip(lower=Q1-1.5*IQR, upper=Q3+1.5*IQR)
```

### **3. Feature Engineering**
- Encoded 5 categorical variables (gender, department, financial aid, parent education, internet access)
- Scaled 4 numerical features (GPA, attendance, study hours, age)
- Dropped identifier column (student_id)

### **4. Class Balancing**
- Applied SMOTE to training data
- Balanced from 613:187 to 613:613 (retained:dropout)
- Maintained original test set distribution for realistic evaluation

### **5. Model Training**
- Algorithm: Random Forest
- Initial: Severe overfitting (95% train, 64% test)
- Optimized: Reduced gap (75% train, 58% test)
- Configuration: 100 trees, max_depth=5, min_samples_split=20, min_samples_leaf=10

### **6. Evaluation**
- Multiple metrics: Accuracy, ROC-AUC, confusion matrix
- Feature importance analysis
- Train-test performance comparison

---
