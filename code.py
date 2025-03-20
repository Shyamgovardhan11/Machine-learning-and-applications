import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load datasets
df_calories = pd.read_csv('/content/calories.csv')  # Update path if needed
df_exercise = pd.read_csv('/content/exercise (2).csv')  # Update path if needed

# Check available columns in both datasets
print("Exercise Data Columns:", df_exercise.columns)
print("Calories Data Columns:", df_calories.columns)

# Find the correct common column name
common_columns = list(set(df_exercise.columns) & set(df_calories.columns))
print("Common Columns Found:", common_columns)

# Merge datasets using the common column if found
if common_columns:
    df = pd.merge(df_exercise, df_calories, on=common_columns[0])
else:
    raise KeyError("No common column found for merging! Check dataset structure.")

# Inspect dataset
print(df.head())
print(df.info())

# Handling missing values
df.dropna(inplace=True)

# Ensure correct column name for Calories Burnt
calorie_column = [col for col in df.columns if 'calorie' in col.lower()]
if calorie_column:
    calorie_column = calorie_column[0]  # Take the first match
else:
    raise KeyError("No calorie-related column found!")

# Define Health Status based on calorie burn threshold
calorie_threshold = df[calorie_column].median()
df['Health_Status'] = np.where(df[calorie_column] >= calorie_threshold, 'Healthy', 'Unhealthy')

# Feature selection
X = df[['Age', 'Height', 'Weight', 'Heart Rate', 'Duration', 'Temperature']]
y = df['Health_Status']

# Encode target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize classification models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000)
}

# Train models
trained_models = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    trained_models[name] = model
    print(f"{name} training complete!\n")

# Evaluate models
model_accuracies = {}
for name, model in trained_models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    model_accuracies[name] = acc
    print(f"Accuracy ({name}):", acc)
    print(f"Classification Report ({name}):\n", classification_report(y_test, y_pred))
    print("-" * 50)

# Visualization using Multiple Techniques
plt.figure(figsize=(12, 6))

# Bar Plot for Model Accuracy Comparison
sns.barplot(x=list(model_accuracies.keys()), y=list(model_accuracies.values()), palette='viridis')
plt.xlabel("Model", fontsize=12)
plt.ylabel("Accuracy Score", fontsize=12)
plt.title("Model Accuracy Comparison", fontsize=14)
plt.ylim(0, 1)
plt.xticks(rotation=30)
plt.show()

# Pie Chart for Health Status Distribution
df['Health_Status'].value_counts().plot.pie(autopct='%1.1f%%', colors=['lightblue', 'orange'], startangle=90)
plt.ylabel('')
plt.title("Health Status Distribution", fontsize=14)
plt.show()

# Pairplot to visualize feature relationships
sns.pairplot(df, hue='Health_Status', diag_kind='kde', palette='husl')
plt.show()

# Heatmap for feature correlation
numerical_df = df.copy()
for col in numerical_df.select_dtypes(include=['object']).columns:
    numerical_df[col] = LabelEncoder().fit_transform(numerical_df[col])

plt.figure(figsize=(8, 6))
sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap", fontsize=14)
plt.show()

# Boxplot for feature distribution
plt.figure(figsize=(10, 6))
sns.boxplot(data=numerical_df.drop(columns=[calorie_column, 'Health_Status']), palette='Set2')
plt.xticks(rotation=30)
plt.title("Feature Distribution Boxplot", fontsize=14)
plt.show()
