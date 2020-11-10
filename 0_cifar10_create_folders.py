import os

root_path = './train/'
root_path_2 = './test/'

os.mkdir(root_path)
os.mkdir(root_path_2)

folders = [
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
for folder in folders:
    os.mkdir(os.path.join(root_path,folder))
    os.mkdir(os.path.join(root_path_2,folder))
