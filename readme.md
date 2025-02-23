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

# Week 4 - In Progress
