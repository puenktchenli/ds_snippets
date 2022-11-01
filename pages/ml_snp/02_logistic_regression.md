[pairing with Andreas](https://github.com/wahumatrum/LiAnd-logistic-regression)
#code_logistic_regression 

## libraries

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
```

#make_classification
	used for generate data using scikit-learn
```python
x, y = make_classification(
	n_samples=100,
	n_features=1,
	n_classes=2,
	n_clusters_per_class=1,
	flip_y=0.03,
	n_informative=1,
	n_redundant=0,
	n_repeated=0
)
```


#confusion_matrix 
> confusiton_matrix(y_test, y_pred)

or using `sns.heatmap()`


#predict_proba 
	to check the actual probability that a data poin belongs to a given class
> logistic_regression.predict_proba(X_test)

#get_params 
>logistic_regression.get_params()

#code_threshold
	there is no built-in function to change default (0.5) threshold for LG in sklearn
```python
proba_array = logistic_regression.predict_proba(X_test)
y_pred_03 = [1 if i[1]>0.3 else 0 for i in proba_array]  #set threshold to 0.3
```


**Remember**: look at the last 10 cost values, to see if they are still descenting


# Viz
#viz_LG

---
## plot the sigmoid function
#sigmoid #viz_LG 

- [scipy.special.expit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.expit.html) is defined as 
  $expit(x) = 1/(1+exp(-x))$ 
  It is the inverse of the logit function.
```python
"""
scipy.special.expit is defined as 
expit(x) = 1/(1+exp(-x)). 
It is the inverse of the logit function.
"""
from scipy.special imort expit

sigmoid_function = expit(df['x'] * lr.coef_[0][0] + lr.intercept_[0]).ravel()

plt.plot(df['x'], sigmoid_function)
plt.scatter(df['x'], df['y'], c=df['y'], cmap='rainbow', edgecolors='b')
```


#viz_confusion_matrix
```python
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])

sns.heatmap(confusion_matrix, annot=True);
```

