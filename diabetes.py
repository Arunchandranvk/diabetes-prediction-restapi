# Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

df = pd.read_csv("diabetes_sample_100.csv")

#  Data Exploration
print("🔍 Missing Values:\n", df.isnull().sum())
print("\n📊 Basic Statistics:\n", df.describe())

print("\n🚨 Anomalies:")
print("Glucose = 0:", (df["Glucose"] == 0).sum())
print("Blood Pressure = 0:", (df["BloodPressure"] == 0).sum())
print("BMI = 0:", (df["BMI"] == 0).sum())



#  Data Preprocessing
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, "scaler.pkl")

#  Exploratory Data Analysis (EDA)
plt.figure(figsize=(15, 10))


# Glucose vs Outcome
plt.subplot(2, 2, 2)
sns.boxplot(x="Outcome", y="Glucose", data=df)
plt.title("Glucose by Outcome")

# Age vs Outcome
plt.subplot(2, 2, 3)
sns.boxplot(x="Outcome", y="Age", data=df)
plt.title("Age by Outcome")

# Correlation Heatmap
plt.subplot(2, 2, 4)
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")

plt.tight_layout()
plt.show()

#  Model Building and Evaluation
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

print("\nModel Performance:\n")
best_model = None
best_score = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    print(f"{name}: Accuracy={acc:.2f}, Precision={prec:.2f}, Recall={rec:.2f}")
    
    if acc > best_score:
        best_score = acc
        best_model = model
        best_model_name = name


joblib.dump(best_model, "model.pkl")
print(f"\n✅ Best model saved: {best_model_name} (Accuracy={best_score:.2f})")

#  Feature Importance (if model is tree-based)
if best_model_name in ["Random Forest", "Decision Tree"]:
    importances = best_model.feature_importances_
    features = df.columns[:-1]
    importance_df = pd.DataFrame({"Feature": features, "Importance": importances})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=importance_df, x="Importance", y="Feature", hue="Feature", legend=False)
    plt.title(f"Feature Importance ({best_model_name})")
    plt.tight_layout()
    plt.show()
