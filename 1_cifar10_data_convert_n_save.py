import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

# Load the CIFAR-10 dataset
cifar10 = tf.keras.datasets.cifar10

# Get test and training data where x are the images and y are the labels
# load dataset
(trainX, trainy), (testX, testy) = cifar10.load_data()

# 224 pixel is used at Mobilenetv2 ,resnet152, ... etc.
# if you want to use other network, please find the input size. 
img_size = 224

# summarize loaded dataset
print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('Test: X=%s, y=%s' % (testX.shape, testy.shape))

CIFAR10_LIST = [
    'airplane', 
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

for idx in range (50000):
    your_image = trainX[idx]
    your_image = cv2.resize(trainX[idx], (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    fig = plt.figure(frameon=False)
    fig.set_size_inches(2,2)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(your_image, aspect='auto')

    fig.savefig('./train/'+str(CIFAR10_LIST[int(trainy[idx])])+'/'+str(idx)+'.jpg',dpi=img_size/2)

    plt.close('all')
    if (idx+1) % 100 ==0:
        print(idx+1,"train images were converted and saved!")

for idx in range (10000):
    your_image = testX[idx]
    your_image = cv2.resize(testX[idx], (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    fig = plt.figure(frameon=False)
    fig.set_size_inches(2,2)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(your_image, aspect='auto')
    # fig.savefig("00000.jpg")#, dpi)

    fig.savefig('./test/'+str(CIFAR10_LIST[int(testy[idx])])+'/'+str(idx)+'.jpg',dpi=img_size/2)

    plt.close('all')
    if (idx+1) % 100 ==0:
        print(idx+1,"test images were converted and saved!")

