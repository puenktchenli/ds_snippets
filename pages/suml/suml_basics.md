---
title: Machine Learning Basics with Scikit-learn
tags: [formatting]
keywords: ml, sklearn
last_updated: Oct 31, 2022
summary: "Basics for Programming Machine Learning"
sidebar: mydoc_sidebar
permalink: ml_basics.html
---



# ML coding with scikit-learn
#code_ml


## Libraries for ML
---
#library_ml  #ml_library 

```python
# sampling data: over/under-sampling
from imblearn.over_sampling import RandomOverSampler 

# Data pre-processing: scaling and splitting
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

# impute, deal with missing date
from sklearn.impute import SimpleImputer

# data label encoding for classification data
from sklearn.preprocessing import LabelEncoder

# pipline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Logistic regression
from sklearn.linear_model import LogisticRegression

# Cross-validation
from sklearn.model_selection import cross_val_score, KFold

# Hyperparameter tunning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import SGDClassifier

# KNN algorithm
from sklearn.neighbors import KNeighborsClassifier

# Naive Bayes
from sklearn.naive_bayes import GaussianNB

# Decision tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

# ensemble methods
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

# Import functionality for cloning a model
from sklearn.base import clone

# model evaluation
from sklearn.metrics import confusion_matrix, accuracy_score, 
from sklearn.metrics import classification_report, f1_score
from sklearn.metrics import roc_curve, recall_score, precision_score


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
```

## Defining X and y
#define_xy
```python
# Defining X and y
features = iris.columns.tolist()
features.remove('species')
X = iris[features]
y = iris.species
```

```python
# Separating the target variable
X = balance_data.values[:, 1:5]
Y = balance_data.values[:, 0]
```

## Split the data
#datasplit #train_test_split 

```python
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3,random_state=1) # by default, test_size = 0.25
```


## General Syntax
```python
from sklearn.module import Model

model = Model()
model.fit(X,y)
predictions =model.predict(X_new)