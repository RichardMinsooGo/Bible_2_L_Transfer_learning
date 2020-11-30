#Importing Libraries
import numpy as np
import cv2
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D,Dense,Input,Conv2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Define network
IMG_SIZE = 224                      # VGG19
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

def net():
    base_model = tf.keras.applications.VGG19(input_shape=IMG_SHAPE,
                                                   include_top=True,
                                                   weights='imagenet')

    # define new empty model
    model = Sequential()

    # add all layers except output from VGG19 to new model
    for layer in base_model.layers[:-3]:
        model.add(layer)

    base_model.trainable = False

    # freeze all weights
    # for layer in model.layers:
    #     layer.trainable = False

    # add dropout layer and new output layer
    model.add(Dropout(0.3))
    model.add(Dense(units=2048, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(100, activation='softmax'))

    model.summary()    
    return model
    
#TPU_init
resolver = tf.distribute.cluster_resolver.TPUClusterResolver('grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)

# This is the TPU initialization code that has to be at the beginning.
tf.tpu.experimental.initialize_tpu_system(resolver)
print("All devices: ", tf.config.list_logical_devices('TPU'))

strategy = tf.distribute.experimental.TPUStrategy(resolver)

with strategy.scope():
    model = net()
    model.compile(loss = 'categorical_crossentropy',optimizer ='adam')

# Load the CIFAR-10 dataset
cifar100 = tf.keras.datasets.cifar100

# load dataset
(X_train, Y_train) , (X_test, Y_test) = cifar100.load_data()

# Shuffle only the training data along axis 0 
def shuffle_train_data(X_train, Y_train): 
    """called after each epoch""" 
    perm = np.random.permutation(len(Y_train)) 
    Xtr_shuf = X_train[perm] 
    Ytr_shuf = Y_train[perm] 
    
    return Xtr_shuf, Ytr_shuf 

train_size = 2000
training_epoch = 1

time0 = time.time()

model.load_weights('./cifar100_VGG19.h5')

for idx in range(training_epoch*25):

    X_shuffled, Y_shuffled = shuffle_train_data(X_train, Y_train)
    (X_train_new, Y_train_new) = X_shuffled[:train_size, ...], Y_shuffled[:train_size, ...] 

    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train_new)

    data = []
    y_batch = []

    for i in range(train_size):
        img = cv2.resize(X_train_new[i], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
        data.append(img)
        y_batch.append(Y_train_new[i])

    data = np.array(data)
    # data = np.float(data)
    y_batch = np.array(y_batch)
    y_batch = tf.keras.utils.to_categorical(y_batch, 100)

    data = data/255.

    # print(data.shape)
    # print(y_batch.shape)    
    # print(data[0,:15,:15,1])
    # print(y_batch[0])

    # Fit function (Really slow. Should do this 100x faster)
    # for i in range(5):
    # time1 = time.time()
    model.fit(data, y_batch, batch_size = 200, epochs=5, verbose=1)
    # time2 = time.time()
    # print(time2-time1)

    if (idx+1)%100 == 0:
        model.save_weights('./cifar100_VGG19.h5', overwrite=True)
        print("Trained epoch :",(idx+1)*5)

time3 = time.time()
print(training_epoch*5," epochs time :",time3-time0)

