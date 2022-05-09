#Importing Libraries
import cv2

import tensorflow as tf
# from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras import layers
from tensorflow.keras import Model, Sequential
from tensorflow.keras import Input  
import matplotlib.pyplot as plt
import numpy as np

# Load the CIFAR-10 dataset
num_classes = 10
EPOCHS = 3

cifar10 = tf.keras.datasets.cifar10

# load dataset
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

# Onehot encode labels
Y_train = tf.keras.utils.to_categorical(Y_train, num_classes)
Y_test  = tf.keras.utils.to_categorical(Y_test, num_classes)

train_size = 50
test_size  = 100
STEPS = int(len(X_train)/train_size)
VAL_STEPS = int(len(X_test)/test_size)

# plot first few images
for i in range(9):
    # define subplot
    plt.subplot(330 + 1 + i)
    # plot raw pixel data
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
    # if you want to invert color, you can use 'gray_r'. this can be used only for MNIST, Fashion MNIST not cifar10
    # pyplot.imshow(trainX[i], cmap=pyplot.get_cmap('gray_r'))
    
# show the figure
plt.show()

# returns batch_size random samples from either training set or validation set
# resizes each image to (224, 244, 3), the native input size for VGG19
#Define network
IMG_SIZE = 224                      # VGG19
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
num_classes = 10                    # cifar10

import sys

class BasicBlock(tf.keras.Model):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, strides=1):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(out_channels, kernel_size=3, strides=strides, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(out_channels, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        
        if strides != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = tf.keras.Sequential([
                layers.Conv2D(self.expansion*out_channels, kernel_size=1, strides=strides, use_bias=False), 
                layers.BatchNormalization()
            ])
        else:
            self.shortcut = lambda x: x
            
    def call(self, x):
        out = tf.keras.activations.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = layers.add([self.shortcut(x), out])
        out = tf.keras.activations.relu(out)
        return out

class BottleNeck(tf.keras.Model):
    expansion = 4
    
    def __init__(self, in_channels, out_channels, strides=1):
        super(BottleNeck, self).__init__()
        self.conv1 = layers.Conv2D(out_channels, kernel_size=1, use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(out_channels, kernel_size=3, strides=strides, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(self.expansion*out_channels, kernel_size=1, use_bias=False)
        self.bn3 = layers.BatchNormalization()
        
        if strides != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = tf.keras.Sequential([
                layers.Conv2D(self.expansion*out_channels, kernel_size=1, strides=strides, use_bias=False), 
                layers.BatchNormalization()
            ])
        else:
            self.shortcut = lambda x: x
            
    def call(self, x):
        out = tf.keras.activations.relu(self.bn1(self.conv1(x)))
        out = tf.keras.activations.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = layers.add([self.shortcut(x), out])
        out = tf.keras.activations.relu(out)
        return out

class BuildResNet(tf.keras.Model):
    def __init__(self, block, num_blocks, num_classes):
        super(BuildResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.layer1 = self._make_layer(block, 64, num_blocks[0], strides=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], strides=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], strides=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], strides=2)
        self.avg_pool2d = layers.AveragePooling2D(pool_size=4)
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(num_classes, activation='softmax')
        
    def call(self, x):
        out = tf.keras.activations.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool2d(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out
    
    def _make_layer(self, block, out_channels, num_blocks, strides):
        stride = [strides] + [1]*(num_blocks-1)
        layer = []
        for s in stride:
            layer += [block(self.in_channels, out_channels, s)]
            self.in_channels = out_channels * block.expansion
        return tf.keras.Sequential(layer)

def ResNet(model_type, num_classes):
    if model_type == 'resnet18':
        return BuildResNet(BasicBlock, [2, 2, 2, 2], num_classes)
    elif model_type == 'resnet34':
        return BuildResNet(BasicBlock, [3, 4, 6, 3], num_classes)
    elif model_type == 'resnet50':
        return BuildResNet(BottleNeck, [3, 4, 6, 3], num_classes)
    elif model_type == 'resnet101':
        return BuildResNet(BottleNeck, [3, 4, 23, 3], num_classes)
    elif model_type == 'resnet152':
        return BuildResNet(BottleNeck, [3, 8, 36, 3], num_classes)
    else:
        sys.exit(ValueError("{:s} is currently not supported.".format(model_type)))

model = ResNet('resnet18', num_classes)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
model_name = 'cifar10_AlexNet'

import os.path
if os.path.isfile(model_name+'.h5'):
    model.load_weights(model_name+'.h5')

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

from tqdm import tqdm, tqdm_notebook, trange

for epoch in range(EPOCHS):
    
    with tqdm_notebook(total=STEPS, desc=f"Train Epoch {epoch+1}") as pbar:    
        train_losses = []
        train_accuracies = []
        
        for s in range(STEPS):
            
            x_batch, y_batch = getBatch(train_size, "train")
            
            out= model.train_on_batch(x_batch, y_batch)
            loss_val = out[0]
            acc      = out[1]*100
            train_losses.append(loss_val)
            train_accuracies.append(acc)
            
            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.4f} ({np.mean(train_losses):.4f}) Acc: {acc:.3f} ({np.mean(train_accuracies):.3f})")

    with tqdm_notebook(total=VAL_STEPS, desc=f"Test_ Epoch {epoch+1}") as pbar:    
        test_losses = []
        test_accuracies = []
        for s in range(VAL_STEPS):
            x_batch_val, y_batch_val = getBatch(test_size, "val")
            evaluation = model.evaluate(x_batch_val, y_batch_val)
            
            loss_val= evaluation[0]
            acc     = evaluation[1]*100
            
            test_losses.append(loss_val)
            test_accuracies.append(acc)

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.4f} ({np.mean(test_losses):.4f}) Acc: {acc:.3f} ({np.mean(test_accuracies):.3f})")
            
model.save_weights(model_name+'.h5', overwrite=True)

# Sample outputs from validation set
LABELS_LIST = "airplane automobile bird cat deer dog frog horse ship truck".split(" ")

n_sample = 8
x_batch_val, y_batch_val = getBatch(n_sample, "val")

for i in range(n_sample):
    import numpy as np
    plt.imshow(x_batch_val[i])
    plt.show()
    print("pred: " + LABELS_LIST[np.argmax(model.predict(x_batch_val[i:i+1]))])
    print("acct: " + LABELS_LIST[np.argmax(y_batch_val[i])])

