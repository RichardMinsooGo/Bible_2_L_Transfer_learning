'''
Data Engineering
'''

'''
D1. Import Libraries for Data Engineering
'''
import tensorflow as tf

'''
D2. Load Cifar10 data / Only for Toy Project
'''

# print(tf.__version__)
cifar10 = tf.keras.datasets.cifar10

# load dataset
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

# Change data type as float. If it is int type, it might cause error
'''
D3. Data Preprocessing
'''
# Normalizing
X_train, X_test = X_train / 255.0, X_test / 255.0

print(Y_train[0:10])
print(X_train.shape)

# One-Hot Encoding
from tensorflow.keras.utils import to_categorical

Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)

'''
D4. EDA(? / Exploratory data analysis)
'''
import matplotlib.pyplot as plt

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

'''
Model Engineering
'''

'''
M1. Import Libraries for Model Engineering
'''

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras import Input
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras import Sequential

'''
M2. Set Hyperparameters
'''

# returns batch_size random samples from either training set or validation set
# resizes each image to (224, 244, 3), the native input size for VGG19
IMG_SIZE = 224                      # VGG19
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
output_dim = 10      # output layer dimensionality = num_classes
EPOCHS = 5
learning_rate = 0.001

'''
M3. Build NN model
'''


class AlexNet(tf.keras.Model):
    def __init__(self, output_dim):
        super(AlexNet, self).__init__()
        self.conv1 = layers.Conv2D(96, kernel_size=11, strides=4, padding='same', activation='relu')
        self.max_pool2d1 = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')
        self.conv2 = layers.Conv2D(256, kernel_size=5, padding='same', activation='relu')
        self.max_pool2d2 = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')
        self.conv3 = layers.Conv2D(384, kernel_size=3, padding='same', activation='relu')
        self.conv4 = layers.Conv2D(384, kernel_size=3, padding='same', activation='relu')
        self.conv5 = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu')
        self.max_pool2d3 = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(4096, activation='relu')
        self.dropout1 = layers.Dropout(0.5)
        self.fc2 = layers.Dense(4096, activation='relu')
        self.dropout2 = layers.Dropout(0.5)
        self.fc3 = layers.Dense(output_dim, activation='softmax')
        
    def call(self, x):
        out = self.max_pool2d1(self.conv1(x))
        out = self.max_pool2d2(self.conv2(out))
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.max_pool2d3(self.conv5(out))
        out = self.flatten(out)
        out = self.dropout1(self.fc1(out))
        out = self.dropout2(self.fc2(out))
        out = self.fc3(out)
        return out

model = AlexNet(output_dim)

'''
M4. Optimizer
'''
# Optimizer can be included at model.compile

'''
M5. Model Compilation - model.compile
'''

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model_name = 'cifar10_AlexNet'

'''
M6. Load trained model
'''

import os.path
if os.path.isfile(model_name+'.h5'):
    model.load_weights(model_name+'.h5')

'''
M7. Define getBatch Function for "model.train_on_batch"
'''
train_size = 50
test_size  = 100
STEPS = int(len(X_train)/train_size)
VAL_STEPS = int(len(X_test)/test_size)

import cv2

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

'''
M8. Define Episode / each step process
'''

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
            
'''
M9. Model evaluation
'''
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
        
'''
M10. Save Model
'''
model.save_weights(model_name+'.h5', overwrite=True)

'''
M11. Sample outputs from validation set
'''
LABELS_LIST = "airplane automobile bird cat deer dog frog horse ship truck".split(" ")

n_sample = 8
x_batch_val, y_batch_val = getBatch(n_sample, "val")

for i in range(n_sample):
    import numpy as np
    plt.imshow(x_batch_val[i])
    plt.show()
    print("pred: " + LABELS_LIST[np.argmax(model.predict(x_batch_val[i:i+1]))])
    print("acct: " + LABELS_LIST[np.argmax(y_batch_val[i])])

