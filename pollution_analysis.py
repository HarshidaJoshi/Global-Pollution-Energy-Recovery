# Global Pollution Analysis and Energy Recovery
# Author: Harshida Joshi

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("Global_Pollution_Analysis.csv")

df.columns = df.columns.str.strip()

print("Dataset Preview")
print(df.head())

# -----------------------------
# Handle Missing Values
# -----------------------------

print("\nMissing Values")
print(df.isnull().sum())

df.fillna(method='ffill', inplace=True)

# -----------------------------
# Descriptive Statistics
# -----------------------------

print("\nDescriptive Statistics")
print(df.describe())

# -----------------------------
# Correlation Heatmap
# -----------------------------

numeric_df = df.select_dtypes(include=['number'])

plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")

plt.title("Correlation Heatmap")

plt.savefig("heatmap.png")
plt.close()

# -----------------------------
# Pollution Trend
# -----------------------------

plt.figure(figsize=(10,6))
sns.histplot(df['aqi_value'], bins=50)

plt.title("AQI Distribution")

plt.savefig("aqi_distribution.png")
plt.close()

# -----------------------------
# Linear Regression
# -----------------------------

features = ['co_aqi_value','ozone_aqi_value','no2_aqi_value','pm2.5_aqi_value']
target = 'aqi_value'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nLinear Regression Results")

print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# -----------------------------
# Pollution Level Classification
# -----------------------------

df['Pollution_Level'] = pd.cut(
    df['aqi_value'],
    bins=[0,50,100,500],
    labels=['Low','Medium','High']
)

le = LabelEncoder()

df['Pollution_Level'] = le.fit_transform(df['Pollution_Level'])

X = df[['co_aqi_value','ozone_aqi_value','no2_aqi_value','pm2.5_aqi_value']]
y = df['Pollution_Level']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

log_model = LogisticRegression()

log_model.fit(X_train, y_train)

y_pred = log_model.predict(X_test)

print("\nLogistic Regression Results")

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# -----------------------------
# Confusion Matrix
# -----------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, cmap="Blues", fmt='d')

plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")
plt.close()

# -----------------------------
# Finished
# -----------------------------

print("\nProject Completed Successfully")
print("Graphs saved: heatmap.png, aqi_distribution.png, confusion_matrix.png")