# Deep Learning Fundamentals

1. [Callbacks](#callbacks)
2. [Datasets](#datasets)
3. [Models & Layers](#models--layers)
4. [Activation Functions](#activation-functions)
5. [Hyper Parameters](#hyper-parameters)
6. [Model Training](#model-training)

## Callbacks

The Callbacks API is used to take actions during training. For this, we create a class which inherits the `tf.keras.callbacks.Callback` and implement one or more of its methods. This can be used to, for example, stop training when a specified metric is met.

We can also use one of the inbuilt classes like `ModelCheckpoint`, `LearningRateScheduler`, `TerminateOnNaN`, `EarlyStopping` etc. We need to pass the object of this class as a parameter when calling the fit method on model. 

```python
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('loss') < 0.4):
            self.model.stop_training = True

callback1 = myCallback()
callback2 = LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))

model.fit(train, test, epochs, callbacks=[callback1, callback2])
```

## Datasets

Some of the popular datasets used for learning are directly available from the `tf.keras.datasets` API. We can load the data by calling the `load_data()` method on appropriate `tf.keras.datasets` class. Some of the available datasets are `mnist` digit classification data, `imdb` review data, `reuters` news classification dataset etc.

## Models & Layers
- Sequential: Defines a model with sequence of layers in the neural network as a list. The output of previous layer is passed as input to next layer.
- Dense: A Dense layer is fully connected layer of neurons. Number of neurons (units) is the required parameter. The other important optional parameters is `activation` which defines the algorithm used by layer.
- Flatten: This layer transforms the input into 1-dimensional array.

## Activation Functions

The activation function decides whether a neuron should be activated or not by calculating the weighted sum and further adding bias to it. The purpose of the activation function is to introduce non-linearity into the output of a neuron.

1. ReLU: Return the element-wise maximum of 0 and the input tensor.
2. Softmax: Takes a list of values and scales these so the sum of all elements will be equal to 1.
3. Sigmoid: For small values(<5), returns a value close to zero and for large values (>5) the result of the function gets close to 1.
4. Linear: This is the default activation type if you do not define it.

## Hyper-Parameters

1. Optimizer: Adam(), SGD(), RMSProp()
2. Loss: Huber(), `binary_cross_entropy`, `categorical_crossentropy`, `sparse_categorical_crossentropy`
3. Metrics: [accuracy], [mae]

## Model Training
We train the model using the fit method.

```python
num_epochs = 30

# Train the model
history = model.fit(train_data, train_labels, 
                    epochs=num_epochs, 
                    validation_data=(test_data, test_labels), 
                    verbose=2
                   )
```

We can set the `verbose` parameter of `model.fit()` to 2 to indicate that we want to print just the results per epoch. Setting it to 1(default) displays a progress bar per epoch, while 0 silences all displays. In production `verbose` 2 is recommended.
