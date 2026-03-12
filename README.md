# Global Pollution Analysis and Energy Recovery

## Objective
The goal of this project is to classify countries into different pollution severity categories (**Low, Medium, High**) based on pollution levels, energy consumption, and other environmental factors.  
Machine Learning techniques are used to analyze environmental indicators and build predictive models for pollution classification.

---

# Phase 1 – Data Preprocessing

## Data Import and Cleaning
- Load the dataset **Global_Pollution_Analysis.csv**
- Handle missing values using appropriate imputation methods
- Detect and handle outliers
- Normalize and standardize features such as:
  - CO2 Emissions
  - Industrial Waste
  - Energy Consumption

## Encoding Categorical Features
Categorical variables such as:
- Country
- Year

are encoded using **LabelEncoder**.

## Feature Engineering
Additional features are created to improve model performance:

- **Energy Consumption per Capita**
- **Yearly Pollution Trends**

Feature scaling is applied to pollution indicators such as:
- Air Pollution
- Water Pollution
- Soil Pollution

---

# Phase 2 – Machine Learning Classification

Three machine learning models are implemented to classify pollution severity.

## 1. Naive Bayes Classifier
- Multinomial Naive Bayes is used
- Classifies countries into:
  - Low Pollution
  - Medium Pollution
  - High Pollution

### Evaluation Metrics
- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1 Score

---

## 2. K-Nearest Neighbors (KNN)

KNN is used to classify pollution levels based on environmental indicators.

### Hyperparameter Tuning
GridSearchCV is used to determine the best value of **K**.

### Evaluation Metrics
- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1 Score

---

## 3. Decision Tree Classifier

Decision Tree is implemented for pollution classification.

### Hyperparameter Tuning
The model is optimized using:

- `max_depth`
- `min_samples_split`

### Evaluation Metrics
- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1 Score

---

# Phase 3 – Model Evaluation and Insights

## Model Comparison
The three classifiers are compared using:

- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1 Score

Visualization techniques are used to interpret the results.

Generated Visualizations:
- Correlation Heatmap
- Pollution Distribution
- Confusion Matrix

---

# Actionable Insights

Based on model predictions:

- Identify countries with **high pollution severity**
- Analyze how pollution affects **energy recovery**
- Recommend strategies for **pollution reduction**
- Suggest improvements in **energy utilization policies**

---

# Project Files
lobal-Pollution-Energy-Recovery
│
├── Global_Pollution_Analysis.csv
├── global_pollution_analysis.py
├── heatmap.png
├── aqi_distribution.png
├── confusion_matrix.png
├── README.md
├── requirements.txt


---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn

---

# Final Deliverables

- Complete Machine Learning Pipeline
- Data Visualization Graphs
- Model Evaluation Reports
- Actionable Environmental Insights