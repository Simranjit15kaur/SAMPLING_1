# Import necessary libraries
import pandas as pd
from collections import Counter  # To handle class distributions
from imblearn.over_sampling import RandomOverSampler, SMOTE  # Oversampling techniques
from imblearn.under_sampling import NearMiss  # Undersampling techniques
from imblearn.under_sampling import TomekLinks  # Tomek Links
from sklearn.model_selection import train_test_split  # Train-test splitting
from sklearn.ensemble import RandomForestClassifier  # Random Forest
from sklearn.linear_model import LogisticRegression  # Logistic Regression
from sklearn.svm import SVC  # Support Vector Classifier
from sklearn.tree import DecisionTreeClassifier  # Decision Tree
from sklearn.neighbors import KNeighborsClassifier  # K-Nearest Neighbors
from sklearn.metrics import accuracy_score, classification_report  # Metrics

# Load dataset
data = pd.read_csv("Creditcard_data.csv")

# Class distribution check
class_count_0, class_count_1 = data['Class'].value_counts()
class_0 = data[data['Class'] == 0]
class_1 = data[data['Class'] == 1]

print('Class 0:', class_0.shape)
print('Class 1:', class_1.shape)

# 1. Random Under-Sampling
class_0_under = class_0.sample(class_count_1)  # Reduce majority class
test_under = pd.concat([class_0_under, class_1], axis=0)
print("\nClass distribution after under-sampling:")
print(test_under['Class'].value_counts())

# 2. Random Over-Sampling
ros = RandomOverSampler(random_state=42)
x = data.drop('Class', axis=1)
y = data['Class']
x_ros, y_ros = ros.fit_resample(x, y)
print('\nClass distribution after Random Over-Sampling:', Counter(y_ros))

# 3. Tomek Links
tl = TomekLinks()
x_tl, y_tl = tl.fit_resample(x, y)
print('\nClass distribution after Tomek Links:', Counter(y_tl))

# 4. SMOTE (Synthetic Minority Oversampling Technique)
smote = SMOTE(random_state=42)
x_smote, y_smote = smote.fit_resample(x, y)
print('\nClass distribution after SMOTE:', Counter(y_smote))

# 5. Near Miss (Undersampling technique)
nm = NearMiss()
x_nm, y_nm = nm.fit_resample(x, y)
print('\nClass distribution after Near Miss:', Counter(y_nm))

# Split the data (stratified sampling)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=42)

# Define ML models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVC": SVC(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
}

# Generate samples with varying proportions
samples = {}
sample_proportions = [0.2, 0.4, 0.6, 0.8, 1.0]  # Proportions to consider
total_samples = len(y_smote)  # Based on SMOTE-balanced dataset

for i, proportion in enumerate(sample_proportions):
    num_samples = int(proportion * total_samples)
    samples[f"Sampling{i+1}"] = (x_smote[:num_samples], y_smote[:num_samples])
    print(f"Sampling{i+1}: {num_samples} samples")

# Initialize results and classification reports
results = {}
classification_reports = {}

# Train and evaluate models
for model_name, model in models.items():
    results[model_name] = []
    classification_reports[model_name] = {}
    
    for sample_name, (X_sample, y_sample) in samples.items():
        X_train, X_test, y_train, y_test = train_test_split(
            X_sample, y_sample, test_size=0.2, random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[model_name].append(accuracy)
        classification_reports[model_name][sample_name] = classification_report(
            y_test, y_pred, output_dict=True
        )

# Create a DataFrame for results
results_df = pd.DataFrame(results, index=[f"Sampling{i+1}" for i in range(len(sample_proportions))])

# Print accuracy results
print("\nAccuracy Results:")
print(results_df)

# Identify the best sampling technique for each model
best_sampling_per_model = results_df.idxmax()
print("\nBest Sampling Technique for Each Model:")
print(best_sampling_per_model)

# Display classification reports
for model_name, reports in classification_reports.items():
    print(f"\nClassification Reports for {model_name}:")
    for sample_name, metrics in reports.items():
        print(f"\n{sample_name}:")
        print(pd.DataFrame(metrics).transpose())
