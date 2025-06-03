# Diabetes Prediction Project Report

## Dataset
- **Source**: `diabetes_sample_100.csv`
- **Rows**: 100
- **Columns**: 9
  - Features: Pregnancies, Glucose, BloodPressure, BMI, Age
  - Target: Outcome (0 = Non-Diabetic, 1 = Diabetic)

---

## 🔍 Exploratory Data Analysis (EDA)

###  Missing Values
No missing values found.

### 🚨 Anomalies Detected:
- Glucose = 0: 0
- BloodPressure = 0: 0
- BMI = 0: 0

### Plots


#### 1. Glucose by Outcome  

![glucose_box](images/glucose_vs_outcome.png)

#### 2. Age by Outcome  
![age_box](images/age_vs_outcome.png)

#### 3. Correlation Heatmap  
![heatmap](images/feature_correlation.png)

---

##  Data Preprocessing
- Features scaled using `StandardScaler`
- Scaler saved as `scaler.pkl`

---

##  Model Training

### Models Tried:
- Logistic Regression
- Decision Tree
- Random Forest

### Evaluation Metrics:


|         Model          | Accuracy | Precision |  Recall  |
|------------------------|----------|-----------|----------|
| Logistic Regression    | 0.50     | 0.44      |   0.44   |
| **Decision Tree**      | **0.75** | **0.75**  | **0.67** |
| Random Forest          | 0.60     | 0.57      |   0.44   |


 **Best model saved**: `Decision Tree` → `model.pkl`


##  Feature Importance (Decision Tree)

![feature_importance](images/feature_importance.png)

---

## Django API

### Endpoints

| Method | Endpoint              | Description                        |
|--------|-----------------------|------------------------------------|
| POST   | `/predict/`           | Predict diabetes                   |
| GET    | `/features/`          | Return feature importance values   |
| GET    | `/health/`            | Check model + API health status    |

### Sample JSON Input

```json
{
  "Pregnancies": 2,
  "Glucose": 120,
  "BloodPressure": 70,
  "BMI": 25.5,
  "Age": 32
}



Screen Recorded Video : https://drive.google.com/file/d/1JgSNhlaF_MFGMkbZUbqkgGxJ46-9EB3D/view?usp=sharing