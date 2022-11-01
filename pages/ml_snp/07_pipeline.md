#pipeline 

![[sk_pipeline.png]]



## Step 1: Preprocessing Pipeline (LogReg)
from NB Diabetes_challage, use logistic regression

```python
# Dropping the unnecessary columns 
df.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1, inplace=True)
df.columns
```

### Categorical vs Numerical Variables

```python
# Creating list for categorical predictors/features 
# (dates are also objects so if you have them in your data you would deal with them first)
cat_features = list(df.columns[df.dtypes==object])
cat_features

# Creating list for numerical predictors/features
# Since 'Survived' is our target variable we will exclude this feature from this list of numerical predictors 
num_features = list(df.columns[df.dtypes!=object])
num_features.remove('Survived')
```

Then we do the Train-Test-Split.
- Deal with the missing values in the numerical and categorical features using `imputer()` 
- Encode all categorical features as a `one-hot`  numeric array.
- Combine both piplines into one called `preprocessor`, using `ColumnTransformer`

```python
#from sklearn.pipeline import Pipeline

# Pipline for numerical features
# Initiating Pipeline and calling one step after another
# each step is built as a list of (key, value)
# key is the name of the processing step
# value is an estimator object (processing step)
num_pipeline = Pipeline([
    ('imputer_num', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler())
])

# Pipeline for categorical features 
cat_pipeline = Pipeline([
    ('imputer_cat', SimpleImputer(strategy='constant', fill_value='missing')),
    ('1hot', OneHotEncoder(handle_unknown='ignore'))
])

```
For quick data cleaning and EDA, it makes a lot of sense to use pandas `get_dummies()`. However, if I plan to transform a categorical column to multiple binary columns for machine learning, it’s better to use `OneHotEncoder().


Use `sklearn.compose.ColumnTransformer

```python
#from sklearn.compose import ColumnTransformer

# Complete pipeline for numerical and categorical features
# 'ColumnTranformer' applies transformers (num_pipeline/ cat_pipeline)
# to specific columns of an array or DataFrame (num_features/cat_features)
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])
```

## Step 2: Predictive Modelling using Pipeline & Grid Search

```python
# a full pipeline with our preprocessor and a LogisticRegression Classifier
pipe_logreg = Pipeline([
    ('preprocessor', preprocessor),
    ('logreg', LogisticRegression(max_iter=1000))
])

# Making predictions on the training set using cross validation as well as calculating the probabilities
# cross_val_predict expects an estimator (model), X, y and nr of cv-splits (cv)
y_train_predicted = cross_val_predict(pipe_logreg, X_train, y_train, cv=5)
```

Then print out the evaluation metics for the model. ([code snippet](obsidian://open?vault=data_science&file=01_%F0%9F%91%A9%F0%9F%8F%BB%E2%80%8D%F0%9F%92%BB_code%2F01_machine_learning%2Fevaluation_matrics))

## Step 3: Optimizing via Grid Search
- Define a parameter space to search for the best parameter combination
- Initiate the grid search via `GridSearchCV`

```python
# Defining parameter space for grid-search. Since we want to access the classifier step (called 'logreg') in our pipeline 
# we have to add 'logreg__' infront of the corresponding hyperparameters. 
param_logreg = {'logreg__penalty':('l1','l2'),
                'logreg__C': [0.001, 0.01, 0.1, 1, 10],
                'logreg__solver': ['liblinear', 'lbfgs', 'sag']
               }

grid_logreg = GridSearchCV(
						   pipe_logreg, 
						   param_grid=param_logreg, 
						   cv=5, scoring='accuracy', 
                           verbose=5, n_jobs=-1
                           )


grid_logreg.fit(X_train, y_train)
```

### Show and save the best model

```python
# Show best parameters
print('Best score:\n{:.2f}'.format(grid_logreg.best_score_))
print("Best parameters:\n{}".format(grid_logreg.best_params_))

# Save best model (including fitted preprocessing steps) as best_model 
best_model = grid_logreg.best_estimator_
best_model
```

## Step 4: Final Evaluation

```python
# Calculating the accuracy, recall and precision for the test set with the optimized model
y_test_predicted = best_model.predict(X_test)

print("Accuracy: {:.2f}".format(accuracy_score(y_test, y_test_predicted)))
print("Recall: {:.2f}".format(recall_score(y_test, y_test_predicted)))
print("Precision: {:.2f}".format(precision_score(y_test, y_test_predicted)))
```

