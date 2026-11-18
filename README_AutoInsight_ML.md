# AutoInsight ML

### An Interactive Data Analysis & Machine Learning Platform

AutoInsight ML is a web-based data analysis and machine learning
application built with **Python**, **Streamlit**, and **Scikit-learn**.
It brings important stages of a typical machine learning workflow into
one interactive workspace --- from exploring and preparing datasets to
training models, evaluating performance, and generating predictions.

The project is designed to make common data science tasks easier to
perform and understand through an interactive interface, visual reports,
and reusable machine learning workflows.

------------------------------------------------------------------------

## Table of Contents

-   [Features](#-features)
-   [Technology Stack](#-technology-stack)
-   [Machine Learning Concepts](#-machine-learning-concepts)
-   [Project Workflow](#-project-workflow)
-   [Project Structure](#-project-structure)
-   [Installation and Setup](#-installation-and-setup)
-   [How to Use](#-how-to-use)
-   [Future Improvements](#-future-improvements)
-   [Author](#-author)

------------------------------------------------------------------------

## 🚀 Features

### 1. Dataset Overview and Data Quality

Upload a CSV dataset and inspect its structure before starting analysis.

-   CSV dataset upload
-   Dataset preview
-   Row and column counts
-   Numerical and categorical column identification
-   Missing-value summary
-   Duplicate-row detection
-   Descriptive statistical summaries
-   Basic data cleaning, including whitespace normalization and
    duplicate removal

### 2. Exploratory Data Analysis (EDA)

Explore data distributions, relationships, and patterns using
interactive visualizations.

**Distribution analysis** - Histograms - Box plots - Violin plots -
Distribution summaries

**Relationship and correlation analysis** - Scatter plots - Feature
relationship exploration - Correlation analysis - Correlation heatmaps

**Outlier analysis** - Interquartile Range (IQR) - Z-score - Outlier
identification and visualization

**Categorical analysis** - Category frequency analysis - Bar charts -
Pie charts and other categorical visualizations

**Missing-value analysis** - Column-wise missing-value counts -
Missing-value percentages - Missingness visualizations

*The exact visualizations available depend on the dataset and the
current application implementation.*

### 3. Missing-Value Handling

Inspect missing data and apply suitable strategies before model
training.

-   Missing-value analysis by column
-   Recommended imputation strategy based on data characteristics
-   Mean imputation
-   Median imputation
-   Most-frequent (mode) imputation
-   Constant-value imputation
-   Removal of rows or columns where appropriate
-   Preview of the processed dataset

### 4. Feature Engineering

Experiment with different feature representations and study their effect
on model performance.

-   **SelectKBest:** Selects a specified number of features using a
    statistical scoring function.
-   **PCA (Principal Component Analysis):** Transforms the original
    features into a smaller set of principal components.
-   Compare model results using original and transformed/selected
    features.

### 5. Machine Learning Model Training

Train and compare machine learning algorithms for classification and
regression problems.

**Classification models** - Logistic Regression - Random Forest
Classifier - Extra Trees Classifier - Decision Tree Classifier -
K-Nearest Neighbors (KNN) - Support Vector Machine (SVM) - Gaussian
Naive Bayes - Gradient Boosting Classifier - AdaBoost Classifier

**Regression models** - Linear Regression - Ridge Regression - Lasso
Regression - Random Forest Regressor - Extra Trees Regressor - Decision
Tree Regressor - K-Nearest Neighbors Regressor - Support Vector
Regression (SVR) - Gradient Boosting Regressor - AdaBoost Regressor

Model availability and configuration may depend on the selected task and
the current application version.

### 6. Model Evaluation and Comparison

Evaluate model performance using metrics appropriate to the selected
machine learning task.

**Classification metrics** - Accuracy - Precision - Recall - F1-score -
ROC-AUC, where supported - Confusion matrix - Classification report -
ROC and Precision-Recall curves, where supported

**Regression metrics** - R² Score - Mean Absolute Error (MAE) - Mean
Squared Error (MSE) - Root Mean Squared Error (RMSE) - Mean Absolute
Percentage Error (MAPE), where supported

The application also provides visual comparisons to help interpret model
results.

### 7. Hyperparameter Tuning

Improve model configurations using search-based optimization methods.

-   Grid Search Cross-Validation
-   Randomized Search Cross-Validation
-   Parameter search for supported models
-   Best-parameter reporting
-   Cross-validation score comparison

### 8. Prediction Centre

Use trained models to generate predictions for new data.

**Manual prediction** - Enter feature values through the application. -
Submit individual or multiple records, depending on the page
implementation. - View generated predictions.

**Batch prediction** - Upload a CSV containing new records. - Generate
predictions for the uploaded dataset. - Review and download prediction
results.

### 9. Model Export

-   Export trained model pipelines.
-   Preserve preprocessing steps together with the estimator when
    supported by the training pipeline.
-   Download model artifacts for reuse in compatible Python
    environments.

------------------------------------------------------------------------

## 🛠️ Technology Stack

  -----------------------------------------------------------------------
  Technology                          Purpose
  ----------------------------------- -----------------------------------
  Python                              Core programming language

  Streamlit                           Interactive web application

  Pandas                              Data loading, manipulation, and
                                      analysis

  NumPy                               Numerical operations

  Scikit-learn                        Preprocessing, machine learning,
                                      evaluation, and tuning

  Plotly                              Interactive data visualizations

  Matplotlib                          Statistical and evaluation
                                      visualizations, where used

  Joblib                              Saving and loading Python model
                                      artifacts
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 🧠 Machine Learning Concepts

  -----------------------------------------------------------------------
  Concept                             Explanation
  ----------------------------------- -----------------------------------
  Exploratory Data Analysis (EDA)     Examining a dataset to understand
                                      its structure, distributions, and
                                      relationships.

  Imputation                          Replacing missing values using
                                      statistical or predefined
                                      strategies.

  Outlier                             An observation that differs
                                      substantially from the general
                                      pattern of the data.

  IQR                                 A statistical measure based on the
                                      first and third quartiles, often
                                      used for outlier detection.

  Z-score                             Measures how many standard
                                      deviations a value is from the
                                      mean.

  Feature Selection                   Choosing a subset of relevant input
                                      features.

  Feature Extraction                  Transforming existing features into
                                      a new representation.

  PCA                                 A dimensionality-reduction
                                      technique that creates principal
                                      components from correlated
                                      features.

  Classification                      Predicting a discrete class or
                                      category.

  Regression                          Predicting a continuous numerical
                                      value.

  Train-Test Split                    Dividing data into training and
                                      testing subsets to evaluate
                                      generalization.

  Cross-Validation                    Evaluating a model across multiple
                                      data splits.

  Hyperparameter Tuning               Searching for model configuration
                                      values that improve validation
                                      performance.

  Pipeline                            A sequence of preprocessing and
                                      modeling steps that can be fitted
                                      and used together.
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 🔄 Project Workflow

A typical workflow in AutoInsight ML is:

1.  **Upload:** Import a dataset in CSV format.
2.  **Inspect:** Review the dataset structure, data types, and data
    quality.
3.  **Explore:** Use EDA visualizations to understand distributions and
    relationships.
4.  **Prepare:** Handle missing values and other data-quality issues.
5.  **Experiment:** Apply feature selection or feature extraction.
6.  **Train:** Select a task and target column, then train available
    models.
7.  **Evaluate:** Compare model performance using suitable metrics.
8.  **Optimize:** Tune hyperparameters for supported models.
9.  **Predict:** Generate predictions for manually entered records or
    uploaded data.
10. **Export:** Download prediction results and supported model
    artifacts.

------------------------------------------------------------------------

## 🗂️ Project Structure

The application is organized into separate areas for the Streamlit
interface, core functionality, and reusable utilities. The exact
filenames may vary by project revision.

``` text
AutoInsight-ML/
│
├── app.py
│
├── pages/
│   ├── Dashboard
│   ├── Exploratory Data Analysis
│   ├── Missing Values
│   ├── Feature Engineering
│   ├── Model Training
│   ├── Model Evaluation
│   ├── Hyperparameter Tuning
│   └── Predictions
│
├── modules/
│   ├── data loading and cleaning
│   ├── data analysis
│   ├── missing-value handling
│   ├── preprocessing
│   ├── feature engineering
│   ├── model training
│   ├── model evaluation
│   ├── hyperparameter tuning
│   └── prediction
│
├── utils/
│   ├── session state
│   ├── helper functions
│   └── chart utilities
│
├── assets/
│
├── requirements.txt
└── README.md
```

------------------------------------------------------------------------

## ⚙️ Installation and Setup

### Prerequisites

-   Python 3.11--3.13 recommended
-   Visual Studio Code or another Python IDE
-   Git (optional, for cloning the repository)

### 1. Clone the repository

Replace the placeholder with the URL of your GitHub repository.

``` bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd AutoInsight-ML
```

Alternatively, download the repository as a ZIP file and extract it.

### 2. Create a virtual environment

``` powershell
python -m venv venv
```

### 3. Install dependencies

On Windows PowerShell, you can call the virtual environment's Python
executable directly. This avoids needing to activate `Activate.ps1` when
PowerShell's script execution policy blocks activation.

``` powershell
.\venv\Scripts\python.exe -m pip install --upgrade pip
```

Install the project dependencies:

``` powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Run the application

Run the following command from the project directory containing
`app.py`:

``` powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit will display a local URL in the terminal, usually:

``` text
http://localhost:8501
```

Open that URL in your browser to use AutoInsight ML.

------------------------------------------------------------------------

## 📖 How to Use

1.  Start the Streamlit application.
2.  Upload a CSV dataset.
3.  Review the dataset overview and data-quality information.
4.  Explore the data using the EDA section.
5.  Inspect missing values and apply an appropriate handling strategy.
6.  Experiment with feature selection or feature extraction.
7.  Select the machine learning task and target column.
8.  Train and compare the available models.
9.  Review evaluation metrics and visualizations.
10. Use hyperparameter tuning for supported models.
11. Generate predictions using manual inputs or a batch CSV.
12. Download prediction outputs or export a trained model when
    available.

------------------------------------------------------------------------

## 🔮 Future Improvements

Potential directions for future development include:

-   Additional feature selection and feature extraction techniques
-   More advanced outlier treatment methods
-   Expanded model libraries and evaluation metrics
-   More extensive hyperparameter search strategies
-   Explainable AI integrations such as SHAP
-   Model loading and experiment management
-   Support for additional input data formats
-   More detailed experiment tracking and reporting

------------------------------------------------------------------------

## 👨‍💻 Author

**Sahil Khedkar**\
Computer Engineering \| Data Science \| Machine Learning \| Data
Engineering

------------------------------------------------------------------------

## 🤝 Contributions

Suggestions, bug reports, and improvements are welcome. Feel free to
open an issue or submit a pull request.

## 📄 License

No license is specified by this README. Add a license file (for example,
an MIT License) if you want to define how others may use, modify, and
distribute this project.

------------------------------------------------------------------------

**AutoInsight ML --- From Raw Data to Machine Learning Predictions.**
