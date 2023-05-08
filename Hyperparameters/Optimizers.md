# Deep Learning Optimizers

## Contents

1. Momentum
2. Nesterov Accelerated Gradient
3. AdaGrad
4. RMSProp
5. Adam
6. Nadam

## Momentum

Momentum optimization cares a great deal about what previous gradients were: at each iteration it substracts the local gradients from the momentum vector, and it updates the weights by adding this momentum vector. The gradiet is used for acceleration, not for speed. To simulate some sort of friction mechanism and prevent momentum from growing too large, the algorithm introduces a new hyperparameter, called the momentum, which must be set between 0(high friction) and 1(no friction). A typical momentum value is 0.9. Momentum optimization escape from plateaus much faster than Gradient Descent.

`optimizer = tf.keras.optimizers.SGD(learning_rate, momentum=0.9)`

## Nesterov Accelerated Gradient

Nesterov Accelerated Gradient measures the gradient of the cost function not at the local position but slightly ahead in the direction of the momentum. This small tweak works because in general the momentum vector will be pointing in the right direction, so it will be slightly more accurate to use the gradient measured a bit farther in that direction rather than the gradient at the original position. NAG is generally faster than regular momentum otimization.

`optimizer = tf.keras.optimizers.SGD(learning_rate, momentum=0.9, nestrov=True)`

## AdaGrad

AdaGrad decays the learning rate, but it does so faster for steep dimensions than for dimensions with general slopes. This is called adaptive learning rate. It helps point the resulting updates more directly toward the global optimum. One addtional benefit is that it requires much less tuning of the learning rate hyperparameter.

AdaGrad frequently performs well for simple quadratic problems, but it often stops too early when training neural networks. The learning rate gets scaled down so much that the algorithm end upt stopping entirely before reaching the global optimum. It should not be used for training deep neural networks.

## RMSProp

RMSProp accumulates only the gradients from the most recent iteration. It does so by using the exponential decay in the first step. The decay rate is typically set to 0.9. 

`optimizer = tf.keras.optimizers.RMSProp(learning_rate, rho=0.9)`

## Adam

Adaptive momentum estimation, combines the ideas of momentum optimization and RMSProp: just like momentum optimization, it keeps track of an exponentially decaying average of past gradients; and just like RMSProp, it keeps track of an exponentially decaying average of past squared gradients. The momentum decay hyperparameter is typically initialized to 0.9, while the scaling decay hyperparameter is often initialized to 0.999. The smoothing term is usually initialized to a tiny number such as 10^-7.

`optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.999)`

## Nadam

Nadam optimization is Adam optimization plus the Nestrov trick, so it will often converge slightly faster than Adam.

---

Adaptive optimization methods (including RMSProp, Adam and Nadam optimization) are often great, converging fast to a good solution. However they can lead to solutions that generalize poorly on some datasets. Sometimes the dataset is allergic to adaptive gradients in that case we should try Nesterov Accelerated Gradient.
