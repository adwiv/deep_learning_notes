# Time Series Analysis and Forcasting

## Contents

1. Time Series
  1.1 Trend
  1.2 Seasonality
  1.3 Noise
  1.4 Autocorrelation
5. Non-Stationary Time Series
6. Naive Forecast 
7. Metrics
8. Moving Average
9. Differencing
10. Smoothing
11. Data Preprocessing For Time Series
12. Training The Model

### Time Series

Time series are data that have a time based variation of input values. There is only one value corresponding to a time - usually a day. The time series generally show variations which can be grouped into the following types:

- **Trend** describes the general tendency of the values to go up or down as time progresses. Given a certain time period, we can see if the graph is following an upward/positive trend, downward/negative trend, or just flat.

- **Seasonality** refers to a recurring pattern at regular time intervals. For instance, hourly temperature might oscillate similarly for 10 consecutive days and you can use that to predict the behavior on the next day.

- **Noise** refers to the day-to-day variations of the data from the standard model. It can play an important role in determining the accuracy of the model.

- **Autocorrelation** means that measurements at a given time step is a function of previous time steps. Another autocorrelation we might encouter is one where it decays predictably after random spikes.

#### Non-Stationary Time Series

It is possible for the time series to break an expected pattern. Big events can alter the trend or seasonal behavior of the data.

### Naive Forecast

As a baseline, we can do a naive forecast where we assume that the next value will be the same as the previous time step. 

### Metrics

1. Mean Square Error: `tensorflow.keras.metrics.mean_squared_error(actual, forecast)`
2. Mean Absolute Error: `tensorflow.keras.metrics.mean_absolute_error(actual, forecaset)`

### Moving Average Smoothing

Smoothing is usually done to help us better see patterns and trends in time series. It generally smooths out the irregular roughness (noise) to see clearer signal. One of the commonly used method for smoothing is moving average.

Moving average takes an average of values over a window. For example, 7 day moving average takes a 7 day window to average out input data. The window is generally trailing, although centered or even forward window can be used as needed.

### Differencing

Differencing is a method of transforming a time series data. It can be used to remove the series dependence on time, so-called temporal dependence. We subtract the identified and fixed variations like trends and seasonality from the data which makes the input data much simpler and better for prediction.

Differencing is performed by substracting the previous observation from the current observation. If the time period between the two observations being subtracted is the same as period of seasonality, it will remove variations both due to trend and seasonality.

`difference(t) = observation(t) - observation(t-N)`

### Data Preprocessing For Time Series

1. Windowing

Here we group consecutive measurements values into one feature and the next measurement will be the label. For example, in hourly measurements, we can use values taken at hours 1 to 11 to predict the value at hour 12. This is done using the `Dataset` class methods. 

2. Flatten the Windows

We want to prepare the windows to be tensors instead of the `Dataset` structure. We can do that by feeding a mapping function to the, this function will be applied to each window and the result will be flattened into a single dataset. 

3. Shuffle The Data

It is a good practice to shuffle the dataset to reduce sequence bias while training the model. The `buffer_size` parameter in shuffle method is required for better shuffling, a good buffer size would be a number equal or greater than the total number of elements.

4. Create Batches For Training 

We should group our windows into batches, we can do this by using the `batch` method on the dataset with `prefetch()` step. This optimizes the execcution time when the model is already training. By specifiying a prefetch `buffer_size` of 1, Tensorflow will prepare the next one batch in advance while the current batch is being consumed by the model. 

```python
# Generate a TF Dataset from the series values
dataset = tf.data.Dataset.from_tensor_slices(series)

# Window the data but only take those with the specified size
dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)

# Flatten the windows by putting its elements in a single batch
dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))

# Create tuples with features and labels 
dataset = dataset.map(lambda window: (window[:-1], window[-1]))

# Shuffle the windows
dataset = dataset.shuffle(shuffle_buffer)

# Create batches of windows
dataset = dataset.batch(batch_size).prefetch(1)

# Now we can call model.fit by passing the dataset as training data

```