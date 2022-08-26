## Credit Card Fraud Transaction Detection
 Credit Card Fraud Transaction Detection is one of the most famous financial Data Science Project. 
 
 #### Steps Involved
 1. Importing the required packages into python environment.
 2. Importing the dataset (Credit Card Fraud Transaction Detection Dataset)
 3. Preprocessing of data as required.
 4. Exploratory Data Analysis (EDA) to extract insights from the dataset.
 5. Training of imbalanced dataset using 3 machine learning classification algorithms and evaluate the trained model using evaluation metrics.
 6. As the dataset was imbalanced, so resample the data using SMOTE technique for better accuracy.
 7. Evaluating the trained model using evaluation metrics.
 
 #### Required Packages
- Pandas -> import pandas as pd
- Numpy -> import numpy as np
- Matplotlib -> import matplotlib.pyplot as plt
- Seaborn -> import seaborn as sns
- Train Test Split -> from sklearn.model_selection import train_test_split
- Classification report and Accuracy score -> from sklearn.metrics import classification_report, accuracy_score  
- Precision and Recall -> from sklearn.metrics import precision_score, recall_score 
- F1 score and corelation co-efficient -> from sklearn.metrics import f1_score, matthews_corrcoef 
- Confusion Matrix -> from sklearn.metrics import confusion_matrix 
- Metrics -> import sklearn.metrics as metrics
- Random Forest Classifier -> from sklearn.ensemble import RandomForestClassifier 
- Logistic Regression -> from sklearn.linear_model import LogisticRegression
- DecisionTree -> from sklearn.tree import DecisionTreeClassifier 
- Smote -> from imblearn.over_sampling import SMOTE
 
 #### About the Data
 The dataset used in this problem is taken from Kaggle.
 - Link to the dataset -> https://drive.google.com/drive/folders/12aygfD-w6-i8pFRLbUgwYr4XoUDEak3p?usp=sharing
 - The dataset consists of features as V1, V2, V3,....., V28 which are principal components obtained by PCA (Principal Component Analysis), the feature "Amount" which
 contains the total money being transacted and the feature "Class" which contains the labels of fraud or non-fraud trancactions in the form of "1" or"0" respectuvely.
 
 #### Data Modeling
 Model is trained before and after resampling of data using three classification algorithms namely Logistic Regression, Random Forest Classifier and Decision Tree Classifier. With these algorithms, the models are trained successfully and each of the models are evaluated to find the most accurate model for this problem. 
 
 From this, I consider that Random Forest classifier is one of the most suitable model as it gives highest accuracy among all the classification algorithms.
