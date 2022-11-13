---
gists:
  - id: 80232b1978fc003a5c9d262b449e64c4
    url: 'https://gist.github.com/80232b1978fc003a5c9d262b449e64c4'
    createdAt: '2022-10-31T21:19:17Z'
    updatedAt: '2022-10-31T21:19:17Z'
    filename: evaluation_matrics.md
    isPublic: false
---


#code_accuracy #classification_report #evaluationmetrics 



### Confusion Metrics using `heatmap()`
#code_confusion_metric 

```python
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score

accuracy_score(y_test, y_pred).round(2)
classification_report(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, cmap='YlGnBu', annot=True, fmt='d', linewidths=.5)
```


### Print `accuracy`, `recall` and `Precision` nicely

```python
# Calculating the accuracy for the LogisticRegression Classifier 
print('Cross validation scores:')
print('-------------------------')
print("Accuracy: {:.2f}".format(accuracy_score(y_train, y_train_predicted)))
print("Recall: {:.2f}".format(recall_score(y_train, y_train_predicted)))
print("Precision: {:.2f}".format(precision_score(y_train, y_train_predicted)))
```

```python
from sklearn.metrics import confusion_matrix

for i,model in enumerate([clf_A,clf_B,clf_C]):
    cm = confusion_matrix(y_test, model.predict(X_test))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # normalize the data

    # view with a heatmap
    plt.figure(i)
    sns.heatmap(cm, annot=True, annot_kws={"size":30}, 
            cmap='Blues', square=True, fmt='.3f')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion matrix for:\n{}'.format(model.__class__.__name__));
```


