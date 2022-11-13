#venv 

```python
pyenv local 3.9.8
python -m venv .venv
source .venv/bin/activate
export HDF5_DIR=/opt/homebrew/Cellar/hdf5/1.12.2
# export HDF5_DIR=/opt/homebrew/Cellar/hdf5/1.12.2_2
```

#tensorflow #tf

```python
tf.config.get_visible_devices()
```

### Setup for TensorBoard

```python
# With this command you can clear any logs from previous runs
# If you want to compare different runs you can skip this cell 
!rm -rf my_logs/

# Define path for new directory 
root_logdir = os.path.join(os.curdir, "my_logs")

# Define function for creating a new folder for each run
def get_run_logdir():
    run_id = time.strftime('run_%d_%m_%Y-%H_%M_%S')
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()

# Create function for using callbacks; "name" should be the name of the model you use
def get_callbacks(name):
    return tf.keras.callbacks.TensorBoard(run_logdir+name, histogram_freq=1)
```


### Normalization layer
#normalization 


```python
normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(X_train))
```

### Build the sequential model

```python
model = tf.keras.Sequential([
    horsepower_normalizer,
    layers.Dense(units=1)
])
model.summary()
model.layer[1].kernel # y=mx+b kernel

```


```python
def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mae',
                  metrics='mse',
                  optimizer=tf.keras.optimizers.Adam(0.001))
    return model
```



### plot 

```python
def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
```


### Evaluate

```python
test_results['dnn_model'] = dnn_model.evaluate(X_test, y_test, verbose=0)
pd.DataFrame(test_results, index=['Mean absolute error [MPG]', 'Mean squared error [MPG]']).T
```

### Make predictions

```python
y_pred = dnn_model.predict(X_test).flatten()

a = plt.axes(aspect='equal')
plt.scatter(y_test, y_pred)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
```


### Save the model
#save_model

```python
dnn_model.save('dnn_model')

# reload
reloaded = tf.keras.model.load_model('dnn_model')

test_results['reloaded'] = reloaded.evaluated(
											  X_test, y_test, verbose=1
											  )
```


## Conclusion

This notebook introduced a few techniques to handle a regression problem. Here are a few more tips that may help:

-   [Mean Squared Error (MSE)](https://www.tensorflow.org/api_docs/python/tf/losses/MeanSquaredError) and [Mean Absolute Error (MAE)](https://www.tensorflow.org/api_docs/python/tf/losses/MeanAbsoluteError) are common loss functions used for regression problems. Mean Absolute Error is less sensitive to outliers. Different loss functions are used for classification problems.
-   Similarly, evaluation metrics used for regression differ from classification.
-   When numeric input data features have values with different ranges, each feature should be scaled independently to the same range.
-   Overfitting is a common problem for DNN models, it wasn't a problem for this tutorial. See the [overfit and underfit](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit) tutorial for more help with this.


