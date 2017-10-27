# From: https://tensorflow.rstudio.com/keras/
# Keras is a high-level neural networks API 
# Keras in R uses the TensorFlow backend engine
# TensorFlow library for numerical computation using data flow graphs

# install keras
## devtools::install_github("rstudio/keras")
# start Keras 
## library(keras)
# install Keras and TensorFlow
## install_keras()
library(keras)

#------------------- BOSTON dataset --------------------------#
# Dataset concerning housing in the area of Boston
# Price of a home is to be predicted
boston <- dataset_boston_housing() # dataset from keras library

X_train <- boston$train$x
Y_train <- boston$train$y
X_test  <- boston$test$x
Y_test  <- boston$test$y
# scale and center the columns of a numeric matrix
X_train <- scale(X_train)
X_test <- scale(X_test)

# construct an empty Sequential model (composed of a linear stack of layers)
model <- keras_model_sequential() 
# use pipe operator to add layers
model %>% 
# add a dense layer with 700 neurons.   
  layer_dense(units = 50, input_shape = c(13)) %>% 
# add an activation defined by a rectified linear unit
  layer_activation("relu") %>%
# add a dense layer with single neuron to serve as the output layer
  layer_dense(units = 1)
# check the layers
model
# compile the model before fitting its parameters or using it for prediction
model %>% 
# loss - objective loss function - costs of an event
# mse - Mean Squared Error - for regression
# more info: https://keras.io/optimizers/
  compile(loss = 'mse', 
          optimizer = optimizer_rmsprop())


# fit the model from the training data
model %>%
# batch_size - number of samples per gradient update  
# epochs - number of times to iterate on a dataset  
  fit(X_train, Y_train, batch_size = 32, epochs = 200, verbose = 1,
      validation_split = 0.1)

# Evaluate the model - predict on test dataset
Y_test_hat <- 
  model %>%
  predict(X_test)

# Mean Squared Error 
sd(as.numeric(Y_test_hat) - Y_test) / sd(Y_test) 
# ~ 0.46

# ------------------ MNIST I
# Handwriting dataset, consisting of small black and white scans of 
# handwritten numeric digits (0-9). 
# The task is to build a classifier that correctly identifies the 
# numeric value from the scan. 
#
# Load this dataset in:
mnist <- dataset_mnist()
X_train <- mnist$train$x
Y_train <- mnist$train$y
X_test <- mnist$test$x
Y_test <- mnist$test$y
# Coerce array n x 28 x 28 into matrix n x 784 (n - number of images)
# scale colours from range [0, 255] into [0,1]
X_train <- array(X_train, 
                 dim = c(dim(X_train)[1], 
                         prod(dim(X_train)[-1]))) / 255
X_test <- array(X_test,
                dim = c(dim(X_test)[1],
                        prod(dim(X_test)[-1]))) / 255

# coerce integer output vector with elements [0,9] into matrix of 10 col
# this is one-hot representation
Y_train <- to_categorical(Y_train, 10)

# construct a neural network from three blocks of identical Dense layers,
# all having 512 nodes, a leaky rectified linear unit, and drop out. 
# These will be followed on the top output layer of 10 nodes and a final
# softmax activation. 

mod <- keras_model_sequential() 

mod %>% 
  layer_dense(units = 512, input_shape = dim(X_train)[2]) %>%
  layer_activation_leaky_relu() %>%
  layer_dropout(rate = 0.25) %>% 
  layer_dense(units = 512) %>%
  layer_activation_leaky_relu() %>%
  layer_dropout(rate = 0.25) %>% 
  layer_dense(units = 512) %>%
  layer_activation_leaky_relu() %>%
  layer_dropout(rate = 0.25) %>% 
  layer_dense(units = 10) %>%
  layer_activation(activation = 'softmax')

# compile the model with the “categorical_crossentropy” loss and 
mod %>% 
  compile(loss = 'categorical_crossentropy',
          optimizer = optimizer_rmsprop())

# fit it on the training data
mod %>%
  fit(X_train, Y_train, batch_size = 32, epochs = 5, verbose = 1,
          validation_split = 0.1)

# make predictions
# instead of 10 columns in output we can get vector of [0-9]
Y_test_hat <- 
mod %>%
  predict_classes(X_test)
# cofuson matrix
table(Y_test, Y_test_hat)
# accuracy
mean(Y_test == Y_test_hat)
# 0.96

# ------------------ MNIST II
mnist <- dataset_mnist() # hand written numbers
# stored as list of 2: train and test datasets
# both are lists of 2: x - predictors, y - target
# x : 3D array with 60k(for train and 10k for test) of images
# each image is 2D array of 28x28 values (px) in range 1-255
# y: vector with values from 0-9
x_train <- mnist$train$x   
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y
# For example first image stored in train is shown below
# transformation (1-x/255) to scale and revert colours 
image.array <- 1-x_train[1,1:28,1:28]/255
image.raster <- as.raster(image.array)
plot(image.raster) # '5'
# Corresponding label:
y_train[1]         # '5'

# Another example, first image stored in test 
image.array <- 1-x_test[1,1:28,1:28]/255
image.raster <- as.raster(image.array)
plot(image.raster) # '7'
# Corresponding label:
y_test[1]         # '7'

# reshape - each of 60k 28 x 28 image represented as a vector
dim(x_train) <- c(nrow(x_train), 784)
dim(x_test) <- c(nrow(x_test), 784)
# rescale - each entry fom [0, 255] into [0, 1] 
x_train <- x_train / 255
x_test <- x_test / 255
# coerce target variable (integer vector) into hot-bit representation 
# that is matrix with 10 columns
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)
# initialize the model
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

history <- model %>% fit(
  x_train, y_train, 
  epochs = 30, batch_size = 128, 
  validation_split = 0.2
)

# Plot accuracy and loss
plot(history)

# evaluate the model
model %>% evaluate(x_test, y_test)
# loss 0.1, accuracy: 0.98

# Generate predictions on new data:
model %>% predict_classes(x_test)


#------------------- BOSTON dataset --------------------------#
# Dataset concerning housing in the area of Boston
# Price of a home is to be predicted
boston <- dataset_boston_housing() # dataset from keras library

X_train <- boston$train$x
Y_train <- boston$train$y
X_test  <- boston$test$x
Y_test  <- boston$test$y
# scale and center the columns of a numeric matrix
X_train <- scale(X_train)
X_test <- scale(X_test)

# construct an empty Sequential model (composed of a linear stack of layers)
model <- keras_model_sequential() 
