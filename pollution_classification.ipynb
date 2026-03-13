# ============================================================
# Global Pollution Analysis and Energy Recovery
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


print("\n========== Global Pollution Analysis ==========\n")

# ============================================================
# Phase 1 – Data Preprocessing
# ============================================================

# Load dataset
df = pd.read_csv("Global_Pollution_Analysis.csv")

print("Dataset Shape:", df.shape)
print(df.head())


# ------------------------------------------------------------
# Handle Missing Values
# ------------------------------------------------------------

df = df.fillna(df.mean(numeric_only=True))

for col in df.select_dtypes(include='object'):
    df[col].fillna(df[col].mode()[0], inplace=True)


# ------------------------------------------------------------
# Encode Categorical Columns
# ------------------------------------------------------------

le = LabelEncoder()

for col in df.select_dtypes(include='object'):
    df[col] = le.fit_transform(df[col])


# ------------------------------------------------------------
# Feature Engineering
# ------------------------------------------------------------

numeric_cols = df.select_dtypes(include=np.number).columns

# Example engineered feature
df["energy_per_capita"] = df[numeric_cols].mean(axis=1)

# Pollution index (combined)
df["pollution_index"] = df[numeric_cols].sum(axis=1)


# ------------------------------------------------------------
# Create Pollution Severity Target
# ------------------------------------------------------------

df["pollution_severity"] = pd.qcut(
    df["pollution_index"],
    q=3,
    labels=["Low", "Medium", "High"]
)

df["pollution_severity"] = LabelEncoder().fit_transform(df["pollution_severity"])


# ------------------------------------------------------------
# Feature Scaling
# ------------------------------------------------------------

scaler = StandardScaler()

X = df.drop("pollution_severity", axis=1)
y = df["pollution_severity"]

X_scaled = scaler.fit_transform(X)


# ------------------------------------------------------------
# Train Test Split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


# ============================================================
# Phase 2 – Machine Learning Models
# ============================================================


# ------------------------------------------------------------
# Naive Bayes
# ------------------------------------------------------------

nb = MultinomialNB()

# MultinomialNB requires positive values
X_train_nb = np.abs(X_train)
X_test_nb = np.abs(X_test)

nb.fit(X_train_nb, y_train)

y_pred_nb = nb.predict(X_test_nb)

print("\nNaive Bayes Results")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))


# ------------------------------------------------------------
# KNN
# ------------------------------------------------------------

param_grid = {"n_neighbors": range(1, 15)}

grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)

knn = grid.best_estimator_

y_pred_knn = knn.predict(X_test)

print("\nKNN Results")
print("Best K:", grid.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))


# ------------------------------------------------------------
# Decision Tree
# ------------------------------------------------------------

param_grid = {
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10]
}

grid_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
grid_dt.fit(X_train, y_train)

dt = grid_dt.best_estimator_

y_pred_dt = dt.predict(X_test)

print("\nDecision Tree Results")
print("Best Parameters:", grid_dt.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))


# ============================================================
# Phase 3 – Visualization & Insights
# ============================================================

# ------------------------------------------------------------
# Confusion Matrix
# ------------------------------------------------------------

cm = confusion_matrix(y_test, y_pred_dt)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, cmap="Blues", fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("confusion_matrix.png")
plt.close()


# ------------------------------------------------------------
# Correlation Heatmap
# ------------------------------------------------------------

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), cmap="coolwarm")

plt.title("Correlation Heatmap")
plt.savefig("heatmap.png")
plt.close()


# ------------------------------------------------------------
# Pollution Distribution
# ------------------------------------------------------------

plt.figure(figsize=(6,4))
sns.histplot(df["pollution_index"], bins=20)

plt.title("Pollution Distribution")
plt.savefig("aqi_distribution.png")
plt.close()


# ============================================================
# Model Comparison
# ============================================================

results = pd.DataFrame({
    "Model": ["Naive Bayes", "KNN", "Decision Tree"],
    "Accuracy": [
        accuracy_score(y_test, y_pred_nb),
        accuracy_score(y_test, y_pred_knn),
        accuracy_score(y_test, y_pred_dt)
    ]
})

print("\nModel Comparison\n")
print(results)


# ============================================================
# Final Output
# ============================================================

print("\nProject Completed Successfully!")
print("Generated files:")
print(" - heatmap.png")
print(" - aqi_distribution.png")
print(" - confusion_matrix.png")