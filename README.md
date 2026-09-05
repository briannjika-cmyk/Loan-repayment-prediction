# Loan-repayment-prediction

A machine learning project that explores and prepares a loan-account dataset and uses **Logistic Regression** to predict customer loan default status.

The project is implemented in a Jupyter Notebook using Python, Pandas, NumPy, Seaborn, Matplotlib, and Scikit-learn.

---

## Project Overview

This project works with a small loan-account dataset containing customer demographic, financial, account, and purchasing information.

The notebook follows a basic machine learning workflow:

1. Load the dataset
2. Inspect the data
3. Identify missing values
4. Remove incomplete records
5. Convert selected columns to numeric data types
6. Explore categorical and numerical variables
7. Clean selected categorical inconsistencies
8. Encode categorical variables using one-hot encoding
9. Remove non-modeling columns
10. Prepare feature (`X`) and target (`y`) datasets
11. Split the data into training and testing sets
12. Train a Logistic Regression classifier
13. Generate predictions
14. Produce a confusion matrix and classification report

---

## Dataset

The original dataset is stored in:

```text
dirty_dataset.csv
```

The dataset contains **50 records and 20 columns** before preprocessing.

### Original Columns

| Column               | Description based on the dataset    |
| -------------------- | ----------------------------------- |
| `customer_id`        | Customer identifier                 |
| `age`                | Customer age                        |
| `income`             | Customer income                     |
| `education`          | Education category                  |
| `marital_status`     | Marital-status category             |
| `credit_score`       | Customer credit score               |
| `loan_amount`        | Loan amount                         |
| `employment_years`   | Years of employment                 |
| `default_status`     | Default-status target variable      |
| `last_payment_date`  | Date of the customer's last payment |
| `phone_number`       | Customer phone number               |
| `email`              | Customer email                      |
| `city`               | Customer city                       |
| `state`              | Customer state                      |
| `zip_code`           | Customer ZIP code                   |
| `product_type`       | Product/loan type                   |
| `purchase_frequency` | Purchase frequency                  |
| `avg_monthly_spend`  | Average monthly spending            |
| `last_login`         | Last account login                  |
| `account_created`    | Account creation date               |

---

## Data Quality Issues

The dataset is intentionally "dirty" and contains several data-quality issues.

The notebook identifies missing values and removes rows containing missing data using:

```python
df = df.dropna()
```

Examples of inconsistencies present in the dataset include:

* Missing values in several columns
* `ERROR` values in fields such as `credit_score`, `employment_years`, and `default_status`
* `HighSchool` and `High School` representing the same education category
* `Divorsed` misspelled instead of `Divorced`
* A `30` value appearing in `purchase_frequency`
* Income values stored as strings rather than consistently numeric values
* Phone-number and email values that are not consistently formatted
* Other potentially questionable values in the raw dataset

The notebook specifically addresses some of these issues rather than performing a complete data-quality correction of every column.

---

## Data Cleaning and Preprocessing

### 1. Missing Values

Missing values are inspected with:

```python
df.isnull().sum()
```

Rows containing missing values are removed:

```python
df = df.dropna()
```

A second missing-value check is performed afterward.

---

### 2. Numeric Conversion

The notebook explicitly converts the following columns to numeric values:

```python
df['credit_score'] = pd.to_numeric(
    df['credit_score'], errors='coerce'
)

df['default_status'] = pd.to_numeric(
    df['default_status'], errors='coerce'
)
```

Later in the preprocessing pipeline:

```python
df['income'] = pd.to_numeric(
    df['income'], errors='coerce'
)

df['employment_years'] = pd.to_numeric(
    df['employment_years'], errors='coerce'
)

df = df.dropna()
```

Using `errors='coerce'` converts values that cannot be interpreted as numbers into missing values, which are subsequently removed.

---

### 3. Marital Status Cleaning

The dataset contains the misspelled category:

```text
Divorsed
```

It is corrected to:

```text
Divorced
```

using:

```python
df['marital_status'] = df['marital_status'].replace(
    'Divorsed',
    'Divorced'
)
```

The categorical variable is then one-hot encoded:

```python
marital = pd.get_dummies(
    df['marital_status'],
    drop_first=True,
    dtype=int
)
```

This produces the encoded columns:

```text
Married
Single
```

---

### 4. Education Cleaning

The dataset contains both:

```text
HighSchool
High School
```

The notebook standardizes `HighSchool` to `High School`:

```python
df['education'] = df['education'].replace(
    'HighSchool',
    'High School'
)
```

One-hot encoding is then applied:

```python
education = pd.get_dummies(
    df['education'],
    drop_first=True,
    dtype=int
)
```

The resulting education features are:

```text
Bachelor
High School
Master
PhD
```

---

### 5. Purchase Frequency Cleaning

The notebook identifies the following purchase-frequency categories:

```text
monthly
weekly
annually
bi-weekly
quarterly
30
```

The value:

```text
30
```

is removed:

```python
df = df[df['purchase_frequency'] != '30']
```

The remaining purchase-frequency categories are one-hot encoded:

```python
p_frequency = pd.get_dummies(
    df['purchase_frequency'],
    drop_first=True,
    dtype=int
)
```

The resulting encoded features are:

```text
bi-weekly
monthly
quarterly
weekly
```

---

## Columns Removed

The following columns are removed before model training:

```python
[
    'customer_id',
    'last_payment_date',
    'phone_number',
    'email',
    'city',
    'state',
    'zip_code',
    'product_type',
    'last_login',
    'account_created',
    'marital_status',
    'purchase_frequency',
    'education'
]
```

The notebook therefore retains the numerical variables and the encoded categorical variables used to construct the modeling dataset.

---

## Final Modeling Features

After preprocessing, the notebook's modeling dataframe contains these columns:

```text
age
income
credit_score
loan_amount
employment_years
default_status
avg_monthly_spend
Bachelor
High School
Master
PhD
Married
Single
bi-weekly
monthly
quarterly
weekly
```

The target variable is:

```text
default_status
```

The predictors are created with:

```python
X = df.drop('default_status', axis=1)
y = df['default_status']
```

This results in **16 predictor variables** and one target variable.

---

## Exploratory Data Analysis

The notebook performs several exploratory operations before modeling.

### Dataset inspection

```python
df.head()
df.describe()
df.info()
```

### Missing-value inspection

```python
df.isnull().sum()
```

### Categorical-value inspection

```python
df['education'].unique()
df['education'].value_counts()

df['marital_status'].value_counts()

df['purchase_frequency'].unique()
df['purchase_frequency'].value_counts()
```

### Visualizations

The notebook creates:

* A count plot for `credit_score`
* A count plot for `default_status`
* A Seaborn pairplot of the dataframe

The notebook does not perform feature-importance analysis.

---

## Machine Learning Model

The project uses **Logistic Regression** from Scikit-learn.

```python
from sklearn.linear_model import LogisticRegression

log = LogisticRegression()
log.fit(X_train, y_train)
```

Logistic Regression is used as a classification model because the notebook treats `default_status` as the prediction target.

---

## Train-Test Split

The dataset is divided into training and testing sets using:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=101
)
```

The configuration used in the notebook is:

| Parameter     | Value |
| ------------- | ----: |
| Test size     |   30% |
| Training size |   70% |
| Random state  |   101 |

No cross-validation is implemented in the notebook.

---

## Model Prediction

Predictions are generated using:

```python
predictions = log.predict(X_test)
```

The notebook then uses:

```python
from sklearn.metrics import classification_report, confusion_matrix
```

to inspect model performance.

---

## Reported Evaluation Output

The executed notebook displays the following confusion matrix:

```text
[[8 0]
 [1 1]]
```

It also displays the following classification report:

```text
              precision    recall  f1-score   support

         0.0       0.89      1.00      0.94         8
         1.0       1.00      0.50      0.67         2

    accuracy                           0.90        10
   macro avg       0.94      0.75      0.80        10
weighted avg       0.91      0.90      0.89        10
```

### Important Evaluation Note

The notebook currently calls:

```python
confusion_matrix(predictions, y_test)
```

and:

```python
classification_report(predictions, y_test)
```

The conventional Scikit-learn order is:

```python
confusion_matrix(y_test, predictions)
```

and:

```python
classification_report(y_test, predictions)
```

Therefore, the displayed evaluation output should **not be presented as a conventional model-performance result without correcting the argument order**.

The README intentionally reports the values that are actually present in the executed notebook rather than reinterpreting them as validated performance metrics.

---

## Final Preprocessed Dataset

Following the transformations implemented in the notebook, the modeling dataframe contains **33 records and 17 columns**:

* 16 predictor columns
* 1 target column (`default_status`)

The final dataframe contains no missing values after the final `dropna()` operation.

---

## Project Structure

A simple repository structure for this project is:

```text
.
├── Loan accounts.ipynb
├── dirty_dataset.csv
└── README.md
```

---

## Technologies Used

* **Python**
* **Jupyter Notebook**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**

---

## How to Run

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Install the required Python packages

```bash
pip install pandas numpy seaborn matplotlib scikit-learn jupyter
```

### 3. Start Jupyter Notebook

```bash
jupyter notebook
```

### 4. Open the notebook

Open:

```text
Loan accounts.ipynb
```

Make sure `dirty_dataset.csv` is located where the notebook can access it.

The notebook loads the dataset using:

```python
df = pd.read_csv('dirty_dataset.csv')
```

### 5. Run the notebook

Execute the cells sequentially to reproduce the data-cleaning, preprocessing, visualization, training, prediction, and evaluation workflow.

---

## Limitations

This project should be viewed as a small educational machine-learning workflow rather than a production-ready credit-risk system.

Based strictly on the notebook and dataset, several limitations are present:

* The dataset contains only 50 original records.
* Rows with missing or coerced values are removed rather than imputed.
* No feature scaling is performed before Logistic Regression.
* No cross-validation is used.
* No hyperparameter tuning is performed.
* The notebook does not compare multiple machine-learning algorithms.
* The notebook does not perform feature-selection analysis.
* The notebook does not calculate feature importance or model coefficients for interpretation.
* Date fields are removed rather than transformed into potentially useful numerical features.
* Several raw data-quality issues are not explicitly corrected.
* The evaluation code uses the prediction array as the first argument to the evaluation functions, which should be corrected before treating the displayed metrics as standard model-performance measurements.

---

## Possible Improvements

Future versions of the project could improve the workflow by:

* Using more data
* Investigating whether invalid values should be corrected, imputed, or removed
* Applying appropriate feature scaling
* Using stratified train-test splitting where appropriate
* Correcting the evaluation argument order
* Adding cross-validation
* Testing additional classification algorithms
* Performing hyperparameter tuning
* Evaluating ROC-AUC and other classification metrics
* Investigating class balance
* Extracting useful information from date fields
* Performing more systematic feature selection
* Saving the trained model for later inference

These are potential improvements and are **not implemented in the current notebook**.

---

## Conclusion

This project demonstrates an end-to-end introductory machine-learning workflow for a loan-account dataset. It covers data inspection, missing-value handling, categorical-data cleaning, one-hot encoding, feature preparation, train-test splitting, Logistic Regression training, prediction, and model evaluation.

The notebook provides a practical example of how a raw dataset containing missing values and inconsistent categorical/numeric entries can be transformed into a dataset suitable for classification.

> **Note:** The project documentation intentionally describes only operations, features, outputs, and limitations that are supported by the supplied notebook and dataset.

