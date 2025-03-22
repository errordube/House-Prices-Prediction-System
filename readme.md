# House Price Prediction System

## Dashboard
<img width="1668" alt="Screenshot 2025-03-22 at 7 52 16 PM" src="https://github.com/user-attachments/assets/3d6b4b7b-6b56-4bb4-bde3-3d66e73717fc" />

## Project Description
The House Price Prediction System is a comprehensive machine learning project designed to predict the sale prices of residential properties based on various features. This project aims to leverage historical data to build a predictive model that can assist buyers, sellers, and real estate professionals in making informed decisions regarding property transactions.

## Objectives
- **Data Analysis**: To explore and analyze the dataset, identifying key features that influence house prices.
- **Data Preprocessing**: To clean and preprocess the data, handling missing values, encoding categorical variables, and normalizing numerical features.
- **Model Development**: To implement and compare various machine learning algorithms, including linear regression, decision trees, random forests, and gradient boosting methods.
- **Model Evaluation**: To evaluate the performance of different models using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared values.
- **Model Optimization**: To fine-tune the selected model using techniques like cross-validation and hyperparameter tuning for improved accuracy.

## Data Sources
The dataset used for this project is derived from the [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview) competition on Kaggle.


## Project Workflow

### **Exploratory Data Analysis (EDA)**
- **Visualizations**: Scatter plots, Z-score analysis for outlier detection.
- **Outlier Handling**: Identified extreme values in `LotArea`, `BsmtFinSF1`, `OverallQual`, etc.

### **Data Preprocessing**
- **Missing Value Handling**:
  - Categorical: `Alley`, `Fence`, `GarageCond` filled with `'No'`.
  - Numerical: `LotFrontage`, `MasVnrArea` filled with `0`.
- **Feature Engineering**:
  - **Dropped Features**: `PoolQC`, `MiscFeature`, `GarageYrBlt`, etc.
  - **Encoding**:
    - **OrdinalEncoder** for ranked categories.
    - **OneHotEncoder** for nominal categories.

### **Model Training & Hyperparameter Tuning**
- **Train-Test Split**: 80% train, 20% test
- **Models Used**:
  - **Random Forest** 
  - **XGBoost** 
  - **Gradient Boosting**
  - **LightGBM** 
  - **CatBoost** 
  - **Ridge Regression**
- **Hyperparameter Tuning**: Used `GridSearchCV` for optimal parameters.

### **Model Evaluation**
- **Metrics Used**: RMSE (Root Mean Squared Error)
- **Final Model Selection**:
  - **Voting Regressor**: Combination of best models with weighted averaging.
  - **Stacking Regressor**: Uses best models as base learners with a meta-estimator (Voting Regressor).

### **Final Prediction & Submission**
- Transformed test data using the pre-processing pipeline.
- Generated predictions using the **Stacking Regressor**.
- **Exponentiated results (`np.exp()`)** to reverse log transformation.
- Saved final predictions to `submission.csv`.


## Results & Observations
- **Voting Regressor + Stacking Model performed the best.**
- **Stacking approach improved performance over individual models.**
- **Feature selection and encoding significantly impacted accuracy.**

## Kaggle 
<img width="1209" alt="Screenshot 2025-03-22 at 7 54 40 PM" src="https://github.com/user-attachments/assets/03d5f39e-cacc-4067-9bca-a5c389ab9c14" />


