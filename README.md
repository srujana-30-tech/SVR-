# Support Vector Regression (SVR) – House Price Prediction

## 📌 Project Overview

This project implements **Support Vector Regression (SVR)** using Python and Scikit-learn to predict house prices.

The dataset contains different input features related to houses, and the target variable is **Price_Lakhs**. The project demonstrates the complete machine learning workflow, including data loading, data exploration, preprocessing, feature scaling, model training, prediction, and model evaluation.

The SVR model uses the **Radial Basis Function (RBF) kernel** to learn the relationship between the input features and house prices.

## 🎯 Objective

The main objectives of this project are:

* Load and explore the house price dataset.
* Check the structure and quality of the data.
* Separate input features and the target variable.
* Split the dataset into training and testing sets.
* Standardize the input features.
* Build an SVR model using the RBF kernel.
* Predict house prices for the test data.
* Evaluate the model using MAE, MSE, and R² Score.
* Compare actual and predicted house prices.

## 🧠 Machine Learning Algorithm

### Support Vector Regression (SVR)

Support Vector Regression is a regression algorithm based on the Support Vector Machine concept.

SVR attempts to find a function that predicts target values while keeping prediction errors within an acceptable margin. It can model both linear and non-linear relationships depending on the kernel used.

In this project, the **RBF (Radial Basis Function) kernel** is used:

```python
SVR(kernel='rbf')
```

### RBF Kernel

RBF stands for **Radial Basis Function**. It is useful for capturing non-linear relationships between input features and the target variable.

## 📂 Dataset

The project uses the following dataset:

```text
Lab 6 SVM Regression.csv
```

The target column is:

```text
Price_Lakhs
```

The remaining columns are used as input features.

The features and target are separated using:

```python
x = df.drop("Price_Lakhs", axis=1)
y = df["Price_Lakhs"]
```

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

## 📦 Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
```

## 🔄 Project Workflow

```text
Load Dataset
      ↓
Explore Dataset
      ↓
Check Dataset Information
      ↓
Check Missing Values
      ↓
Separate Features and Target
      ↓
Train-Test Split
      ↓
Feature Scaling
      ↓
Train SVR Model
      ↓
Make Predictions
      ↓
Evaluate Model
      ↓
Compare Actual vs Predicted Values
```

## 1. Import Required Libraries

The required Python libraries are imported for data manipulation, visualization, preprocessing, model building, and evaluation.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
```

## 2. Load the Dataset

The dataset is loaded using Pandas:

```python
df = pd.read_csv("Lab 6 SVM Regression.csv")
```

The dataset is stored in a DataFrame called `df`.

## 3. Explore the Dataset

The first few records are displayed using:

```python
print("Dataset Preview")
print(df.head())
```

Statistical information is obtained using:

```python
df.describe(include='all')
```

Dataset information is checked using:

```python
df.info()
```

The notebook also checks:

```python
print(df.columns)
print(df.shape)
print(df.dtypes)
```

These commands help understand the dataset's columns, dimensions, and data types.

## 4. Check Missing Values

Missing values are checked using:

```python
print(df.isnull().sum())
```

This calculates the number of missing values in each column.

## 5. Separate Features and Target

The target variable is `Price_Lakhs`.

```python
x = df.drop("Price_Lakhs", axis=1)
y = df["Price_Lakhs"]
```

Where:

* `x` represents the input features.
* `y` represents the target variable.

## 6. Train-Test Split

The dataset is divided into training and testing data:

```python
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=42
)
```

### Parameters

* `test_size=0.3` means 30% of the data is used for testing.
* The remaining 70% is used for training.
* `random_state=42` makes the split reproducible.

## 7. Feature Scaling

SVR is sensitive to feature scales, so `StandardScaler` is used:

```python
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
```

The scaler is fitted only on the training data and then applied to the test data.

This prevents information from the test set from being used during training.

## 8. Create the SVR Model

The Support Vector Regression model is created using the RBF kernel:

```python
model = SVR(kernel='rbf')
```

The RBF kernel allows the model to capture non-linear relationships between the input features and house prices.

## 9. Train the Model

The model is trained using:

```python
model.fit(x_train, y_train)
```

The model learns the relationship between the input features and house prices from the training data.

## 10. Make Predictions

Predictions are generated using the test data:

```python
y_pred = model.predict(x_test)
```

The variable `y_pred` contains the predicted house prices.

## 11. Model Evaluation

The model is evaluated using three regression metrics:

### Mean Absolute Error (MAE)

```python
mae = mean_absolute_error(y_test, y_pred)
```

MAE measures the average absolute difference between actual and predicted values.

A lower MAE indicates smaller average prediction errors.

### Mean Squared Error (MSE)

```python
mse = mean_squared_error(y_test, y_pred)
```

MSE calculates the average squared difference between actual and predicted values.

Larger errors have a greater effect because the errors are squared.

A lower MSE indicates smaller prediction errors.

### R² Score

```python
score = r2_score(y_test, y_pred)
```

R², or the coefficient of determination, measures how well the model explains the variation in the target variable.

The result is displayed using:

```python
print("\nr2_score: ", score)
```

## 12. Display Evaluation Results

The evaluation metrics are printed using:

```python
print("\nMean Absolute Error: ", mae)
print("\nMean Squared Error: ", mse)
print("\nr2_score: ", score)
```

The exact values depend on the dataset and the results obtained when the notebook is executed.

## 13. Compare Actual and Predicted Values

A DataFrame is created to compare actual and predicted house prices:

```python
result = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})
```

The first 10 results can be displayed using:

```python
print(result.head(10))
```

This provides a direct comparison between the actual house prices and the prices predicted by the SVR model.

## 📊 Model Evaluation Metrics

| Metric   | Purpose                                                      |
| -------- | ------------------------------------------------------------ |
| MAE      | Measures the average absolute prediction error               |
| MSE      | Measures the average squared prediction error                |
| R² Score | Measures how well the model explains variation in the target |

## 🔑 Important Concepts

### Regression

Regression is a supervised machine learning technique used to predict continuous numerical values.

In this project, the continuous value being predicted is:

```text
Price_Lakhs
```

### Supervised Learning

The model learns from input features and their corresponding known target values.

```text
Input Features → SVR Model → Price Prediction
```

### Feature Scaling

Feature scaling places input features on a comparable scale.

This is particularly important for SVR because the algorithm is sensitive to feature magnitudes.

### SVR

Support Vector Regression is the regression version of Support Vector Machines and can model non-linear relationships using kernels.

### RBF Kernel

The Radial Basis Function kernel allows SVR to model non-linear relationships between the features and target variable.

## 📁 Project Structure

```text
SVR-Regression/
│
├── SVR.ipynb
├── Lab 6 SVM Regression.csv
└── README.md
```

## ▶️ How to Run the Project

### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
```

### Step 2: Open the Project Folder

```bash
cd SVR-Regression
```

### Step 3: Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Step 4: Start Jupyter Notebook

```bash
jupyter notebook
```

### Step 5: Open the Notebook

Open:

```text
SVR.ipynb
```

### Step 6: Keep the Dataset in the Same Folder

Make sure the following file is available in the same folder as the notebook:

```text
Lab 6 SVM Regression.csv
```

### Step 7: Run All Cells

Run the notebook cells from top to bottom to load the dataset, preprocess the data, train the SVR model, generate predictions, and evaluate the model.

## 📌 Key Takeaways

* This project demonstrates **Support Vector Regression (SVR)**.
* The project predicts house prices using the target variable `Price_Lakhs`.
* The dataset is divided into training and testing sets using a 70:30 split.
* `StandardScaler` is used for feature scaling.
* SVR with an **RBF kernel** is used for regression.
* Predictions are generated for the test dataset.
* Model performance is evaluated using **MAE, MSE, and R² Score**.
* Actual and predicted values are compared using a Pandas DataFrame.

## 🚀 Future Improvements

The model can be further improved by:

* Tuning SVR hyperparameters such as `C`, `gamma`, and `epsilon`.
* Comparing different SVR kernels.
* Performing cross-validation.
* Comparing SVR with other regression algorithms.
* Performing additional exploratory data analysis.
* Adding actual-vs-predicted visualization.
* Comparing model performance using multiple evaluation metrics.

## 👩‍💻 Project Type

**Machine Learning – Regression**

**Algorithm:** Support Vector Regression (SVR)

**Kernel:** RBF (Radial Basis Function)

**Target Variable:** Price_Lakhs
Author 
Srujana
