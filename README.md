![image](https://github.com/user-attachments/assets/f48bbbc6-b0a5-41a1-a5b8-135b71a39d13)

<h1 align="center">ML Regression Project for CSGO Player Score Prediction</h1>

<h4><u>Objective :</u></h4> 
Build a machine learning model to perform <b>Regression</b> based on a real dataset from Kaggle. Project done in python on jupyter notebook.

<h4><u>Notes :</u></h4> 

Train models and evaluate mainly based on `R2` parameter (above `0.8` is a good model).

# Table of Contents

- [A. Pipeline Overview](#apipeline-overview)
  - [A.1. Data Import](#a1-data-import--from-kaggle)
  - [A.2. Exploratory Data Analysis (EDA) & Data Visualization](#a2-exploratory-data-analysis-eda--data-visualization)
  - [A.3. Data Preprocessing](#a3-data-preprocessing)
  - [A.5. Model Training](#a5-model-training)
  - [A.6. Model Evaluation](#a6-model-evaluation)
  - [A.7. Model Deployment](#a7-model-deployment--not-included)
- [B. Report File extracted by ydata-profiling](#breport-file-extracted-by-ydata-profiling)
- [C. Prediction Model Summary](#c-the-file-summarizing-the-prediction-models-includes-the-adjusted-r-squared-r-squared-rmse-and-time-taken-parameters-and-prediction-execution-time-based-on-this-dataset)
  - [C.1. List of trained models sorted](#c1-this-list-of-trained-models-sorted-by-fast-execution-time-and-high-r-squared-parameter)
  - [C.2. Top 5 as horizontal bar charts](#c2-top-5-as-horizontal-bar-charts)
  - [C.3. Scatter plot (Actual value and Prediction)](#c3-regression-model-evaluation-chart-1-scatter-plot)
  - [C.4. Bar plot (Absolute error of the first 50 samples)](#c4-regression-model-evaluation-chart-2-bar-plot-(absolute-error-of-the-first-50-samples))
  - [C.5. Residual plot](#c5-residual-plot)
  - [C.6. Histogram of residuals](#c6-histogram-of-residuals)
- [D. Results](#d-results)


# A.Pipeline-Overview
1. **Data Import** : from Kaggle
2. **Exploratory Data Analysis (EDA) & Data Visualization**  
   - Check for missing values  
   - Analyze variable distributions  
   - Compute correlation matrix
   - Create a report of dataset from ydata_profiling
3. **Data Preprocessing**  
   - Data encryption :
     - Numerical features : Standardization by StandardScaler and then using median and IQR to remove outliers to increase model performance by RobustScaler
     - Categorical Nominal feature : using OneHotEncoder
     - Categorical Ordinal features : determining the orders then using OrdinalEncoder
     - Categorical Boolean features : supposing 2 values of each series ​​are ordered (just random !) and also use OrdinalEncoder
   - Filling in missing value (Nan) by SimpleImputer
   - Train/Test split
   - (Bonus) GridSearchCV to find the best hyperparameter
5. **Model Training** 
   - Linear Regression  
   - Random Forest
   - (Bonus) LazyPredict to show performances of 30-40 differents models
6. **Model Evaluation** 
   - MAE, MSE, Coefficient of determination (R2)
7. **Model Deployment** : not included

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

# C. The file summarizing the prediction models includes the Adjusted R-Squared, R-Squared, RMSE and Time Taken parameters and prediction execution time based on this dataset. 

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



# D. Results
- All models were trained and evaluated to identify the best-performing one for this regression task.
- Reusability on other datasets thanks to calling the pickle library to store the models.






