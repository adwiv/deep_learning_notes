# Time Series Analysis and Forcasting

## Contents

1. Trend
2. Seasonality
3. Noise
4. Autocorrelation
5. Non-Stationary Time Series
6. Naive Forecast 
7. Metrics
8. Moving Average
9. Differencing
10. Smoothing
11. Data Preprocessing For Time Series
12. Training The Model


## Trend

The trend describes the general tendency of the values to go up or down as time progresses. Given a certain time period, we can see if the graph is following an upward/positive trend, downward/negative trend, or just flat.

## Seasonality

Seasonality refers to a recurring pattern at regular time intervals. For instance, hourly temperature might oscillate similarly for 10 consecutive days and you can use that to predict the behavior on the next day.

## Noise

In real-life time series data have some noise in it. They can play an important role in determining the accuracy of the model.

## Autocorrelation

Time series can also be autocorrelated. This means that measurements at a given time step is a function of previous time steps. Another autocorrelation we might encouter is one where it decays predictably after random spikes.

## Non-Stationary Time Series

It is possible for the time series to break an expected pattern. Big events can alter the trend or seasonal behavior of the data.

## Naive Forecast

As a baseline, we can do a naive forecast where we assume that the next value will be the same as the previous time step. 

## Metrics

1. Mean Square Error: `tensorflow.keras.metrics.mean_squared_error(actual, forecast)`
2. Mean Absolute Error: `tensorflow.keras.metrics.mean_absolute_error(actual, forecaset)`

## Moving Average

Moving average sums up a series of time steps and the average will be the prediction for the next time step. For example, the average of the measurements at time steps 1 to 10 will be the forecast for the time step 11, then the average for the time steps 2 to 11 will be the the forecast for time step 12, and so on. Moving average does not anticipate trend of seasonality. Huge spikes in series can cause big deviations.

## Differencing

Differencing is a method of transforming a time series data. It can be used to remove the series dependence on time, so-called temporal dependence. This includes structures like trends and seasonality.

Differencing is performed by substracting the previous observation from the current observation.

`difference(t) = observation(t) - observation(t-1)`

## Smoothing

Smoothing is usually done to help us better see patterns, trends in time series. It generally smooths out the irregular roughness to see clearer signal. Smoothing tools smooths a numeric variable of one or more time series using centered, forward, and backward moving average, as well as an adaptive method based on local linear regression.

## Data Preprocessing For Time Series

1. Windowing

Here we group consecutive measurements values into one feature and the next measurement will be the label. For example, in hourly measurements, we can use values taken at hours 1 to 11 to predict the value at hour 12. 

2. Flatten the Windows

We wan to prepare the windows to be tensors instead of the `Dataset` structure. We can do that by feeding a mapping function to the, this function will be applied to each window and the result will be flattened into a single dataset. 

3. Shuffle The Data

It is a good practice to shuffle the dataset to reduce sequence bias while training the model. The `buffer_size` parameter in shuffle method is required for better shuffling, a good buffer size would be a number equal or greater than the total number of elements.

4. Create Batches For Training 

We should group our windows into batches, we can do this by using the `batch` method on the dataset with `prefetch()` step. This optimizes the execcution time when the model is already training. By specifiying a prefetch `buffer_size` of 1, Tensorflow will prepare the next one batch in advance while the current batch is being consumed by the model. 

## Training Model 

While traing a Deep Neural Network model for predicting forecast we can leverage the power of learning rate scheduler callback. This will allow us to dynamically set the learning rate based on the epoch number during training.

`tensorflow.keras.callbacks.LearningRateScheduler()`


