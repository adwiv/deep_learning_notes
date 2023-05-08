# Computer Vision

## Contents

1. Convolutional Neural Network
2. Image Data Generator
3. Visualizing Intermediate Representation
4. Data Augmentation
5. Transfer learning

## Convolutional Neural Network

CNN takes an image and scan it over the entire image. By changing the underlying pixels based on the formula withing the matrix, we can perform operations like edge detection. We can think of a convnet as an information disitllation pipeline wherein each layer filters out the most usefule features.

`tf.keras.layers.Conv2D()`

The convolutional network is followed by `MaxPool2D` layer, which is designed to compress the image, while maintaining the content of the features that were highlighted by the convolution. By specify (2,2) for the MaxPooling, the effect is to quarter tthe size of the image. The idea is that it creates a 2x2 array of pixels and picks the biggest one.

`tf.keras.layers.MaxPool2D()`

## Image Data Generator

Image Data Generator automatically labels the images according to the directory names and structure. For example, if we have a 'training' directory containing 'horses' directory and a 'humans' one. The Image Data Generator will label the images 'horses' and 'humans'.

```
{
        from tf.keras.preprocessing.image import ImageDataGenerator

        train_datagen = ImageDataGenerator(rescale=1/255)

        train_generator = train_datagen.flow_from_directory(
                            './path/',
                            target_size=(),
                            batch_size=(),
                            class_mode='')
        )
    }

```
Training will be performed on the `train_generator` object by passing it to the `model.fit()`.

## Visualizing Intermediate Representation

To get a feel for what kind of features our CNN has learned, one fun thing to do is to visualize how an input gets transfromed as it goes through the model. 

We can pick a random image from the training set, and then generate a figure where each row is the output of a layer, and each image in the row is a specific filter in that output feature map. Return this cell to generate intermediate representations of a variety of training images.

The representations downstream start highlighting what the network pays attention to, and they show fewer and fewer features being 'activated'; most are set to zero. This is called representation sparsity and is a key feature of deep learning. These representations carry increasingly less information about the original pixels of the image, but increasingly refined information about the class of the image. 

## Data Augmentation

Data augmentation increases the amount of training data by modifiying the existing training data's properties. For example, in image data, we can apply different preprocessing techniques such as rotate, flip, shrear or zoom on the existing image. This way the model would see more variety in the images during training so it will infer better on new, previously unseed data. 

```
{
        train_datagen = ImageDataGenerator(
                                           rotation_range=,
                                           width_shift_range=,
                                           height_shift_range=,
                                           shear_range=,
                                           zoome_range=,
                                           horiontal_flip=
                                           fill_mode='')
        )
    }
```
## Transfer Learning 

In transfer learning we use pre-trained model to achieve a good results even with a small training dataset. It leverages the trained layers of an existing model and adding it to our own layers to fit our needs.

- Steps:

1. Set the input shape to fit your application.
2. Pick and freeze the layer to take advantage of the features it has learned already.
3. Add dense layers which we will train.

- Example: InceptionV3
```
{
        from tensorflow.keras.applications.inception_v3 import InceptionV3
        from tensoflow.keras import layers

        local_weight_file = './path/to/the/downloaded/weights'

        pre_trained_model = InceptionV3(input_shape= (),
                                        include_top=,
                                        weights=)

        pre_trained_model.load_weights(local_weight_file)

        for layer in pre_trained_model.layers:
            layer.trainable = False
    }
```







