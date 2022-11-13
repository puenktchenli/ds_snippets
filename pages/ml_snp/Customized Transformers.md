#feature_transform  #transformerMinin #baseestimator

Sometimes you might want to transform your features in a very specific way, which is not implemented in scikit-learn yet. In those cases you can create your very own custom transformers. In order to work seamlessly with everything scikit-learn provides you need to create a class and implement the three methods `.fit()`, `.transform()` and `.fit_transform()`.      
Two useful base classes on which you can construct your personal transformer can be imported with the following command:

```python
from sklearn.base import BaseEstimator, TransformerMixin
```