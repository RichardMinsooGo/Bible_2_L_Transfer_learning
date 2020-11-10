import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras import Model, Sequential

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

img_size = 224
batch_size = 100
steps_per_epoch = int(50000/batch_size)
n_epoch = 5

# Initialising the CNN
model = Sequential()

# Step 1 - Convolution
model.add(Conv2D(32, (3, 3), input_shape = (img_size, img_size, 3), activation = 'relu'))

# Step 2 - Pooling
model.add(MaxPool2D(pool_size = (2, 2)))

# Adding a second convolutional layer
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPool2D(pool_size = (2, 2)))

# Step 3 - Flattening
model.add(Flatten())

# Step 4 - Full connection
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 10, activation = 'softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('./train',
                                                 target_size = (img_size, img_size),
                                                 batch_size = batch_size,
                                                 # class_mode = 'sparse')
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('./test',
                                            target_size = (img_size, img_size),
                                            batch_size = batch_size,
                                            # class_mode = 'sparse')
                                            class_mode = 'categorical')

model.fit_generator(training_set,steps_per_epoch = steps_per_epoch, epochs = n_epoch) #,
                         # validation_data = test_set, validation_steps = 2000)



