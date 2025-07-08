![image](https://github.com/user-attachments/assets/f48bbbc6-b0a5-41a1-a5b8-135b71a39d13)

<h1 align="center">ML Regression Project for CSGO Player Score Prediction</h1>

<h4><u>Objective :</u></h4> 
Build a machine learning model to perform <b>Regression</b> based on a real dataset from Kaggle. Project done in python on jupyter notebook.

<h4><u>Notes :</u></h4> 

Train models and evaluate mainly based on `R2` parameter (above `0.8` is a good model). 

# Table of Contents

- [A. Pipeline Overview](#apipeline-overview)
- [B. Report File extracted by ydata-profiling](#breport-file-extracted-by-ydata-profiling)
- [C. Summarizing](#c-summarizing)
  - [C.1. List of trained models sorted](#c1-this-list-of-trained-models-sorted-by-fast-execution-time-and-high-r-squared-parameter)
  - [C.2. Top 5 as horizontal bar charts](#c2-top-5-as-horizontal-bar-charts)
  - [C.3. Scatter plot (Actual value and Prediction)](#c3-regression-model-evaluation-chart-1-scatter-plot)
  - [C.4. Bar plot (Absolute error of the first 50 samples)](#c4-regression-model-evaluation-chart-2-bar-plot-absolute-error-of-the-first-50-samples)
  - [C.5. Residual plot](#c5-regression-model-evaluation-chart-3-residual-plot)
  - [C.6. Histogram of residuals](#c6-regression-model-evaluation-chart-4-histogram-of-residuals)
  - [C.7. Quickly check other models through MAE, MSE, RMSE, R2](#c7-quickly-check-other-models)
- [D. Results](#d-results)


# A.Pipeline-Overview
1. **Data Import**

   * Dataset loaded from Kaggle using `pandas`.

2. **Exploratory Data Analysis (EDA)**

   * Checked dataset shape, data types, null values, and unique values.
   * Performed basic descriptive statistics for numerical and categorical features.

3. **Data Preprocessing**

   * Split into `train/test` sets with `random_state=42` to preserve class distribution.
   * Handled missing values using `SafeMissForest` (based on the `MissForest` function in the `missforest` library).
   * Encoded categorical features (`map`, `result`) with `OneHotEncoder`.
   * Scaled numerical features using `StandardScaler`.
   * Applied Chi-squared test for feature selection based on statistical significance.

4. **Model Training**

   * Trained a **Linear Regression** model using `sklearn`.

5. **Model Evaluation**

   * Used metrics: **MAE**, **MSE**, **R²**.
   * Visualized prediction vs actual, residuals, and absolute errors.

6. **Model Deployment**

   * Saved and reloaded the model using `pickle` for future use.


# B.Report-File-extracted-by-ydata-profiling

![image](https://github.com/user-attachments/assets/fc9da2e8-8173-42ba-9132-5937e4b18467)
![image](https://github.com/user-attachments/assets/28dfa042-7af5-4670-bcc4-e56f781b2260)
![image](https://github.com/user-attachments/assets/14072eca-b6cb-43e3-b06d-fb5159874809)
![image](https://github.com/user-attachments/assets/ec3ef25a-2aca-4c06-9185-d1772c83dfd9)
![image](https://github.com/user-attachments/assets/5082ece1-5de2-450e-9df4-3e11c9ef8b2c)
![image](https://github.com/user-attachments/assets/9330c57f-19b7-4528-89d3-b09606c88063)
![image](https://github.com/user-attachments/assets/897b3ef4-6110-4f19-b783-add715c26a0b)
![image](https://github.com/user-attachments/assets/8bcde6cf-4d64-4112-a187-1663788df473)
![image](https://github.com/user-attachments/assets/b6b2e19f-2336-41ec-8f75-a0227762ab67)
![image](https://github.com/user-attachments/assets/9cc01f0d-2671-4edf-99d1-bbe760d2b7fd)
![image](https://github.com/user-attachments/assets/a2277850-8e56-4be3-a099-ebe23eb1bfaa)
![image](https://github.com/user-attachments/assets/4ac2d010-b5eb-4f33-b823-cfbb1425f5b0)
![image](https://github.com/user-attachments/assets/919ef5aa-0a2e-4480-80ad-6b615076183d)
![image](https://github.com/user-attachments/assets/3a7f5fee-3cba-47fe-a282-b4768f129d8c)
![image](https://github.com/user-attachments/assets/236efd5b-ea79-41c8-b1cf-956f4c34d7e5)

# C. Summarizing 

#### C.1. This list of trained models sorted by fast execution time and high R-Squared parameter
![image](https://github.com/user-attachments/assets/f3da6eb0-3c7c-4892-9ef7-5a68fa826ffb)

#### C.2. Top 5 as horizontal bar charts
![image](https://github.com/user-attachments/assets/893571c3-061a-4eab-b5b6-c768edd5011a)

#### C.3. Regression model evaluation chart 1: Scatter plot
![image](https://github.com/user-attachments/assets/9fd8143e-88b6-4031-bf50-2e75d407053d)

#### C.4. Regression model evaluation chart 2: Bar plot (Absolute error of the first 50 samples)
![image](https://github.com/user-attachments/assets/75bbbca6-ec70-4b4f-a801-e92f8c1d13f7)

#### C.5. Regression model evaluation chart 3: Residual plot
![image](https://github.com/user-attachments/assets/87fa3f85-439f-4fab-99b9-9aae79881814)

#### C.6. Regression model evaluation chart 4: Histogram of residuals
![image](https://github.com/user-attachments/assets/596ad803-7c6f-4e00-9098-7214fc3923a2)

#### C.7. Quickly check other models
![image](https://github.com/user-attachments/assets/e8f05fd6-9d30-409d-9a55-2b5c491355e1)



# D. Results
- All models were trained and evaluated to identify the best-performing one for this regression task.
- Reusability on other datasets thanks to calling the pickle library to store the models.






