# Customer Churn Prediction

## Table of Contents
- [Project Overview](#project-overview)
- [Data Sources](#data-sources)
- [Tools Used](#tools-used)
- [Data Cleaning](#data-cleaning)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Decision Tree Classifer](#decision-tree-classifer)
- [Random Forest Classifer](#random-forest-classifer)
- [Support Vector Machine](#support-vector-machine)
- [Logistic Regression](#logistic-regression)
- [K-Nearest Neighbors](#k-nearest-neighbors)
- [Gradient Boosting Classifier](#gradient-boosting-classifier)
- [Power BI Dashboard](#power-bi-dashboard)
- [Recommendations](#recommendations)
- [References](#references)

### Project Overview

This project aims to predict customer churn using Machine Learning techniques. The dataset is imbalanced (90% non-churn, 10% churn), so strategies like resampling and feature engineering were applied to improve performance. Models such as Logistic Regression, Random Forest, and XGBoost were evaluated using metrics like Precision, Recall, and F1-Score.


### Data Sources

The dataset for this project was obtained from Primo Academy Hackathon (customer_data.csv, churn_data.csv, historical_price_data.csv) . It includes customer demographics, account information, and usage details.

### Tools Used

- Python
- Pandas
- Scikit-Learn
- Matplotlib
- Seaborn
- Power BI

### Data Cleaning

The dataset underwent thorough preprocessing to ensure quality and consistency:
1. Handling Missing Values: Identified and filled/nullified missing entries appropriately.
2. Removing Duplicates: Checked and eliminated duplicate records to maintain data integrity.
3. Encoding Categorical Variables: Transformed non-numeric features into numerical representations using techniques like One-Hot Encoding and Label Encoding.
4. Feature Scaling: Standardized numerical features for model compatibility.
5. Outlier Detection: Detected and addressed outliers to prevent skewed model performance.

### Exploratory Data Analysis

EDA was conducted to understand the structure and relationships within the data, which included:
- Summary Statistics: Generated descriptive statistics to understand distributions, central tendencies, and variance of numerical features.
- Correlation Analysis: Identified correlations between features and the target variable (churn) using heatmaps and pair plots.
- Visualizations: Used bar plots, histograms, and box plots to visualize distributions and relationships of key features like age, account length, and usage.
- Class Distribution: Analyzed the imbalance in the target variable (churn) to determine the need for resampling techniques.

### Decision Tree Classifer

~~~jupyter
model_smote = DecisionTreeClassifier(criterion = 'gini', random_state=100,max_depth = 6,min_samples_leaf = 6)
model_smote.fit(Xr_train,yr_train)
y_pred =model_smote.predict(Xr_test)
~~~

### Random Forest Classifer

~~~jupyter
model_smote_rf = RandomForestClassifier(n_estimators = 100, criterion = 'gini', random_state=100,max_depth = 6,min_samples_leaf = 6)
model_smote_rf.fit(Xr_train,yr_train)
y_pred_rf =model_smote_rf.predict(Xr_test)
~~~

### Support Vector Machine

~~~jupyter
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=100)
svm_model.fit(Xr_train, yr_train)
y_pred_svm = svm_model.predict(Xr_test)
~~~

### Logistic Regression

~~~jupyter
log_reg = LogisticRegression(solver='liblinear', penalty='l2', C=1.0, class_weight='balanced', random_state=100)
log_reg.fit(Xr_train, yr_train)
y_pred_log = log_reg.predict(Xr_test)
~~~

### K-Nearest Neighbors

~~~jupyter
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(Xr_train, yr_train)
y_pred_knn = knn_model.predict(Xr_test)
~~~

### Gradient Boosting Classifier

~~~jupyter
clf = GradientBoostingClassifier(random_state=42)
clf.fit(Xr_train, yr_train)
y_pred_gb = clf.predict(Xr_test)
~~~

### Results / Findings

After evaluating multiple models, the following key insights were obtained:

- Model Performance:

   - Gradient Boosting Classifier (GBC) delivered the highest performance, with an F1-score of 0.91, providing a good balance between precision and recall.
   - KNN had a slightly lower accuracy but still performed well with an AUC-ROC of 0.83, making it suitable for risk scoring.
   - Random Forest showed strong performance, particularly in terms of recall, highlighting its ability to detect potential churners.
   - Logistic Regression and SVM performed less effectively compared to the others, with KNN struggling because of its simpler linear approach and SVM being computationally expensive.

#### Class Imbalance:

   - Addressing class imbalance with SMOTE (Synthetic Minority Over-sampling Technique) significantly improved recall, especially in models like Random Forest and Logistic Regression, which had trouble identifying churners in an imbalanced dataset.

#### Feature Importance:

   - Key features influencing churn predictions included monthly charges, account tenure, contract type, and tech support. Customers with month-to-month contracts and higher monthly charges had a higher likelihood of churning.
   - Customer tenure was a critical indicator, with shorter tenure customers more likely to churn, especially those with low tech support engagement.

### Power BI Dashboard

A Power BI dashboard was created to visualize key insights from the customer churn analysis.

![Dashboard](https://github.com/user-attachments/assets/f800989e-0094-46c6-9799-6c26d3c4ec28)

### Recommendations

1. Target High-Risk Customers: Focus retention efforts on customers with short tenure and month-to-month contracts.
2. Incentivize Long-Term Contracts: Offer discounts to encourage customers to switch to longer-term plans.
3. Reduce High Monthly Charges: Provide personalized pricing plans or bundle offers for customers with high charges.
4. Improve Customer Support: Enhance tech support to address dissatisfaction and proactively engage customers.
5. Monitor Key Metrics: Track churn indicators like tenure and customer activity to intervene early.

### References

- Scikit-Learn Documentation – Machine Learning Models
- SMOTE for Class Imbalance – Chawla, N. V., et al. (2002). “Synthetic Minority Over-sampling Technique.”
- Ensemble Methods – Friedman, J. H. (2001). “Greedy Function Approximation: A Gradient Boosting Machine.”
- Python Libraries – Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn.
