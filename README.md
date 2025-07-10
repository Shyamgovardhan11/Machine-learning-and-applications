# 🔥 Calorie Burnt Prediction Using Ensemble Learning

This project is a Machine Learning-based system to **predict calorie expenditure** and classify individuals as **Healthy or Unhealthy** using **ensemble learning techniques**. Developed with Python and popular ML libraries, this project uses feature-rich physiological and workout data to train accurate classification models.

## 🚀 Project Highlights

- 📊 **Ensemble Models Implemented**:
  - Random Forest
  - Gradient Boosting (Highest accuracy: **98.7%**)
  - Logistic Regression
- ⚙️ **Techniques Used**:
  - Data Preprocessing (handling nulls, feature scaling, label encoding)
  - Feature Engineering (based on median calorie burn)
  - Model Evaluation (classification report, confusion matrix, accuracy)
  - Hyperparameter Tuning
  - Data Visualization with **Matplotlib** and **Seaborn**

---

## 📁 Dataset Used

The project uses two CSV files:
- `calories.csv`: Contains calorie burn information
- `exercise.csv`: Contains exercise and physiological metrics

These datasets are merged using a common key for unified analysis.

---

## 🧠 ML Workflow

### 1. Data Preparation
- Merging datasets
- Cleaning missing values
- Feature selection: Age, Height, Weight, Duration, Heart Rate, Temperature
- Label encoding: Healthy (1) / Unhealthy (0)

### 2. Visualization
- Correlation Heatmap
- Box Plots, Bar Graphs
- Pair Plots
- Pie Charts

### 3. Model Training
- Splitting into 80% training and 20% testing sets
- Scaling features using `StandardScaler`
- Training three ML models
- Evaluating accuracy, precision, recall, and F1-score

### 4. Best Model
- **Gradient Boosting** achieved highest performance with **98.7% accuracy**

---

## 📈 Sample Results

![Classification Report](assets/classification_report.png)
![Heatmap](assets/heatmap.png)

> *(Add screenshots or results from your `Code With Results.pdf` here in an `assets/` folder in your repo.)*

---

## 🛠️ Tech Stack

- **Language**: Python
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **Models**: `RandomForestClassifier`, `GradientBoostingClassifier`, `LogisticRegression`

---

## 📂 How to Run

```bash
# Clone the repo
git clone https://github.com/yourusername/calorie-burnt-prediction.git
cd calorie-burnt-prediction

# Install dependencies
pip install -r requirements.txt

# Run the main file
python calorie_prediction.py
