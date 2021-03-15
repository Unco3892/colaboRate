#########################################################
### Instructions on how to run Rstudio on Google Colab###
#########################################################

# Beside Colab, another option is to use https://gradient.paperspace.com/free-gpu ,however,
# the you can only run an operation for 6 hours

#-------------------------------------------------------------------------------
# install the packages
#install.packages(c("keras", "tensorflow", "reticulate"))
library(tensorflow)
library(keras)
library(reticulate)

# if necessary you can run this otherwise it's not needed
#reticulate::use_python("/usr/lib/python3.7")
py_config()
# if you want to check the memory and the GPU that you have, run the command 
#`nvidia-smi`in the terminal

#-------------------------------------------------------------------------------
# Clone your GitHub repo from the `terminal` tab and you can manually open a \
# different R project if you would like to. for example For a public repo run
# git clone https://github.com/Unco3892/colaboRate
# git config --global user.email "you@example.com"
# git config --global user.name "Your Name"

# Note: it is recommended to use `renv` to restore all the packages more quickly

#-------------------------------------------------------------------------------
# Running a test model
mnist <-dataset_mnist() # it will tell you whether you would like to select miniconda, and you have to
#reply no

x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# tf$debugging$set_log_device_placement(TRUE)

# reshape
dim(x_train) <- c(nrow(x_train), 784)
dim(x_test) <- c(nrow(x_test), 784)
# rescale
x_train <- x_train / 255
x_test <- x_test / 255

y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

model <- keras_model_sequential()
model %>%
  layer_dense(units = 256,
              activation = "relu",
              input_shape = c(784)) %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = "softmax")

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(),
  metrics = c("accuracy")
)


history <- model %>% fit(
  x_train,
  y_train,
  epochs = 30,
  batch_size = 256,
  validation_split = 0.2
)

#-------------------------------------------------------------------------------
# google driving mounting data to be added