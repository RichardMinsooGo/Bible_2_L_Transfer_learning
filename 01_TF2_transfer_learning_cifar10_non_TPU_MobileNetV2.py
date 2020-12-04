#Importing Libraries
import numpy as np
import cv2

import matplotlib.pyplot as plt
import time

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout

IMG_SIZE = 224                      # MobileNetV2
# Also you can choose below option
# IMG_SIZE = 160                      # MobileNetV2
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
num_classes = 10                    # cifar10

# 사전 훈련된 모델 VGG19 에서 기본 모델을 생성합니다.
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

base_model.summary()


base_model.trainable = False

# freeze all weights
# for layer in model.layers:
#     layer.trainable = False

# add dropout layer and new output layer
model = tf.keras.Sequential([
                          base_model,
                          tf.keras.layers.GlobalAveragePooling2D(),
                          tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)])

model.summary()

# import sys
# sys.exit()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
model_name = 'cifar10_MobileNetV2'

# Load the CIFAR-10 dataset
cifar10 = tf.keras.datasets.cifar10

# load dataset
(X_train, Y_train) , (X_test, Y_test) = cifar10.load_data()

# Onehot encode labels
Y_train = tf.keras.utils.to_categorical(Y_train, num_classes)
Y_test = tf.keras.utils.to_categorical(Y_test, num_classes)

train_size = 250
test_size = 500
training_epoch = 3
STEPS = int(50000/train_size)

import os.path
if os.path.isfile(model_name+'.h5'):
    model.load_weights(model_name+'.h5')

# returns batch_size random samples from either training set or validation set
# resizes each image to (224, 244, 3), the native input size for VGG19
def getBatch(batch_size, train_or_val='train'):
    x_batch = []
    y_batch = []
    if train_or_val == 'train':
        idx = np.random.randint(0, len(X_train), (batch_size))

        for i in idx:
            img = cv2.resize(X_train[i], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
            x_batch.append(img)
            y_batch.append(Y_train[i])
    elif train_or_val == 'val':
        idx = np.random.randint(0, len(X_test), (batch_size))

        for i in idx:
            img = cv2.resize(X_test[i], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
            x_batch.append(img)
            y_batch.append(Y_test[i]) 
    else:
        print("error, please specify train or val")

    x_batch = np.array(x_batch)
    y_batch = np.array(y_batch)
    return x_batch, y_batch

for e in range(training_epoch):
    train_loss = 0
    train_acc = 0

    for s in range(STEPS):
        x_batch, y_batch = getBatch(train_size, "train")
        out = model.train_on_batch(x_batch, y_batch)
        train_loss += out[0]
        train_acc += out[1]

    print(f"Epoch: {e+1}\nTraining Loss = {train_loss / STEPS}\tTraining Acc = {train_acc / STEPS}")

    x_batch_val, y_batch_val = getBatch(test_size, "val")
    eval = model.evaluate(x_batch_val, y_batch_val)
    print(f"Validation loss: {eval[0]}\tValidation Acc: {eval[1]}\n")
    
model.save_weights(model_name+'.h5', overwrite=True)

# Sample outputs from validation set
LABELS_LIST = "airplane automobile bird cat deer dog frog horse ship truck".split(" ")

x_batch_val, y_batch_val = getBatch(10, "val")

for i in range(10):
    import numpy as np
    plt.imshow(x_batch_val[i])
    plt.show()
    print("pred: " + LABELS_LIST[np.argmax(model.predict(x_batch_val[i:i+1]))])
    print("acct: " + LABELS_LIST[np.argmax(y_batch_val[i])])

