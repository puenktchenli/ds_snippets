#decision_tree #code #gini #entropy



### classification tree
```python
from sklearn.tree import DecisionTreeClassifier
```

### Regression tree
```python
from sklearn.tree import DecisionTreeRegressor
```

### using `Gini Impurity` 

```python 
from sklearn.tree import DecisionTreeClassifier

clf_gini = DecisionTreeClassifier(
								  criterion = "gini", 
								  max_depth=3, 
								  min_samples_leaf=5
								  )

clf_gini.fit(X_train, y_train)

```

### using `Entropy`

```python
clf_entropy = DecisionTreeClassifier(
									 criterion = "entropy",
									 max_depth = 3, 
									 min_samples_leaf = 5
									 )

```

### Pruning tree
#pruning 

 handling overfitting

-  `max_leaf_nodes
-  `max_depth
-  `min_samples_split
-  `max_features



### Viz: Tree
#plot_tree #savefig


#### use `sklearn
```python
from sklearn.tree import plot_tree

fig = plt.figure(figsize=(25,20))

dectree_plot = plot_tree(dec_tree,feature_names=['Production Cost'], filled=True)

# to export the graph
plt.savefig('decision_tree')

```

![](blob:vscode-webview://17619sdcq9cf9q6mkouil2ebphveg9ubjnf1u699camt86doaqfh/caea5f56-70dc-47e4-80cf-56fce53ce6e4)

![[Screenshot 2022-10-15 at 16.15.01.png]]

### use `dtreeviz`

[source](https://mathdatasimplified.com/2022/10/14/dtreeviz-visualize-and-interpret-a-decision-tree-model-2/)

```python
from sklearn.datasets import load_wine
from sklearn import tree
from dtreeviz.trees import dtreeviz

wine=load_wine()
classifier = tree.DecicisionTreeClassifier(max_depth=2)
classifier.fit(wine.data, wine.target)

viz = dtreeviz(classifier,
			  wine.data,
			  wine.target,
			  target_name="wine_type",
			  feature_names=wine.feature_names)

```
![[Screenshot 2022-10-15 at 17.22.49.png]]