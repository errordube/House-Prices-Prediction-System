# House Price Prediction System

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

# Week 1 - Project Setup

## Tasks
1. **Setting up Git/GitHub**  
   - Initialize a Git repository for version control.
   - Link the project to a GitHub repository for remote tracking and collaboration.

2. **Setting up Python, VSCode, and Jupyter Notebook**  
   - Install Python and configure it for the project.  
   - Set up VSCode as the primary IDE with required extensions.  
   - Install Jupyter Notebook for exploratory data analysis.

3. **Setting up AWS (Pending)**  
   - Configure AWS services for data storage and computational resources (to be completed later).

4. **Setting up Virtual Environment using Pipenv**  
   - Install and initialize a virtual environment using `pipenv` to manage dependencies effectively.  
   - Add essential libraries and packages required for the project.

# Week 2 - Data Understanding and Data Processing

## Tasks

1. **Importing Libraries**
   - I started with importing all the relevant libraries like pandas, numpy, seaborn and matplotlib.
   - There are several libraries you will find in some chunks which were needed at that time.

2. **Understanding the data**
   - Used train_df.info() to check the number of features, data types, and missing values.
   - Used train_df.describe() to analyze summary statistics for numerical columns.

3. **Exploring the Target Variable**
   - I then used train_df['SalePrice'].hist() to see a distribution of the target variable.
   - This showed a right-skewed distribution.

4. **Identifying Numeric and Categorical Features**
   - Extracted numeric and categorical by including their dtypes as int64, float64 and object.

5. **Checking for coorelations**
   - A correlation matrix was used to analyze relationships between numerical features.
   - This helped in identifying highly correlated variables.
   - GarageYrBlt and YearBuilt are highly correlated, GrLivArea and TotRMsAbvGrd too are highly correlated.

6. **Significance Test**
   - Both Anova and Chi-Square tests was conducted, for now i am going with Anova test results.
   - I Only considered statistcally significant variables and dropped non-significant one. 

7. **Handling Missing Values**
   - The percentage of missing values in each column was computed.
   - Numeric missing values were imputed using KNN Imputation.
   - Categorical missing values were imputed using Mode Imputation. 
   - Features with more than 50% missing values were dropped.

8. **Feature Transformation**
   - The SalePrice variable was transformed using log transformation to normalize skewness.
   - The 'Id' column was removed as it does not contribute to prediction.

# Week 3 - Feature Engineering and Feature Selection

## Tasks

1. **New features were created to enhance model performance**
   - Age: Difference between YrSold and YearBuilt.
   - TotalBath: Total number of bathrooms, weighting half-baths as 0.5.
   - TotalRooms: Sum of total rooms above ground and bedrooms.
   - TotalFlrSF: Total floor square footage.
   - LotFrontage_Bin: Binned lot frontage into Low, Medium, and High using pd.qcut().
   - NeighborhoodQuality: Mean SalePrice by neighborhood mapped to all houses.
   - IsHighSeason: Flag indicating if the sale occurred between April-August.

2. **Feature Scaling and Encoding**
   - RobustScaler was applied to numerical features to handle outliers
   - One-Hot Encoding was applied to categorical features

3. **Feature Selection Using Random Forest**
   - A Random Forest Regressor was trained to determine feature importance
   - The top 30 features were selected for the final dataset

# Week 4 - Supervised Model Building

## Tasks

1. **Model Training with Random Forest Regressor**
   - A Random Forest Regressor was initialized with 100 estimators and trained on the dataset.
   - A Gradient Boosting Regressor was initialized and trained on the dataset.
   - Model Training with XGBoost

2. **Model Predictions**

   - Predictions were generated on the training set.

3. **Model Evaluation**

   - The model was evaluated using Mean Squared Error (MSE) and R² Score:

   - Results: MSE: 0.003 ; R² Score: 0.984 (indicating high model performance on the training data)
   - MSE: 0.007; R² Score: 0.956 (GBR)
   - MSE: 0.003; R² Score: 0.980 (XGBoost)

4. **Cross-Validation (CV) to Assess Model Stability**

   - Cross-validation with 5 folds was used to check model generalization.

   - Results: Mean MSE: 0.02; MSE Scores across folds: [0.016, 0.020, 0.018, 0.016, 0.019]
   - Mean MSE: 0.016 (GBR)
   - Mean MSE: 0.016 (XGBoost)

5. **Cross-Validation for R² Score**

   - R² Score was calculated using cross-validation
   - Results: Mean R²: 0.89; R² Scores across folds: [0.88, 0.88, 0.87, 0.88, 0.87]
   - Mean R² Score: 0.901 (GBR)
   - Mean R² Score: 0.898 (XGBoost)

High R² score on the training set indicates strong model performance.
Cross-validation results suggest good generalization capability.
Low MSE confirms that the model makes accurate predictions.
Gradient Boosting Regressor achieved a Mean R² of 0.901, while XGBoost achieved 0.898.
XGBoost had the lowest MSE (0.003) on training, indicating strong learning capability.


# Week 5 - Hyperparameter Tuning, Model Evaluation and Model Selection

1. **Hyperparameter Tuning**

   - We tuned Random Forest, Gradient Boosting, and XGBoost models using RandomizedSearchCV to find the best hyperparameters efficiently.

2. **Model Training and Evaluation**

   - Gradient Boosting & XGBoost performed the best with a Mean MSE of 0.016 and a Mean R² Score of 0.901.
   - Random Forest had slightly worse performance with a higher Mean MSE (0.018) and a lower Mean R² Score (0.887).

3. **Final Model Selection**

   - Since XGBoost had the lowest Mean MSE and highest R², it was selected as the final model.
   - Results were then saved as final_predictions.csv for submission or further evaluation.