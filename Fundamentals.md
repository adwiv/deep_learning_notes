# Deep Learning Fundamentals

## Contents

1. Callbacks
2. Datasets
3. Models 
4. Activation Functions
5. Hyper-Parameters

### Callbacks

The Callbacks API is used to stop training when a specified metric is met.
This can be achieved by creating a class which inherits the `tf.keras.callbacks.Callback`.

```
{
        class myCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs={}):
                if (logs.get('loss') < 0.4):
                    self.model.stop_training = True
    }
```

We can call pass the object of this class as a parameter when calling the fit method on model. 
`model.fit(train, test, epochs, callbacks=[callbacks])`

### Datasets

Most popular datasets are directly available from the `tf.keras.datasets` API. We can load the data using `data = tf.keras.datasets.name_of_data` and can call the `load_data()` method.

Normalizing the image data is an important step when solving computer vision problems, this can be achieved by `train_data = train_data / 255.0`.

### Models

- Sequential: Defines a sequence of layers in the neural network.
- Flatten: Takes the data and transforms it into 1-dimensional array.
- Dense: Adds a layer of neurons.

### Activation Functions

The activation function decides whether a neuron should be activated or not by calculating the weighted sum and further adding bias to it. The purpose of the activation function is to introduce non-linearity into the output of a neuron.

1. ReLU: Return the element-wise maximum of 0 and the input tensor.
2. Softmax: Takes a list of values and scales these so the sum of all elements will be equal to 1.
3. Sigmoid: For small values(<5), returns a value close to zero and for large values (>5) the result of the function gets close to 1.

### Hyper-Parameters

1. Optimizer: Adam(), SGD(), RMSProp()
2. Loss: Huber(), `cross_entropy`, `categorical_crossentropy`, `sparse_categorical_crossentropy`
3. Metrics: [accuracy], [mae]


