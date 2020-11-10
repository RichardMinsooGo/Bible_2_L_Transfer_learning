# example of loading the fashion mnist dataset
import os

root_path = './train/'
root_path_2 = './test/'
n_classes = 10

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
  
# Function to rename multiple files 
def main():
    for idx in range(n_classes):
        for count, filename in enumerate(os.listdir("./train/"+str(CIFAR10_LIST[idx])+"/")): 
            dst = str(CIFAR10_LIST[idx])+"_"+str(count) + ".jpg"
            src ="./train/"+str(CIFAR10_LIST[idx])+"/"+ filename 
            dst ="./train/"+str(CIFAR10_LIST[idx])+"/"+ dst 

            # rename() function will 
            # rename all the files 
            os.rename(src, dst) 


    for idx in range(n_classes):
        for count, filename in enumerate(os.listdir("./test/"+str(CIFAR10_LIST[idx])+"/")): 
            dst = str(CIFAR10_LIST[idx])+"_"+str(count) + ".jpg"
            src ="./test/"+str(CIFAR10_LIST[idx])+"/"+ filename 
            dst ="./test/"+str(CIFAR10_LIST[idx])+"/"+ dst 

            # rename() function will 
            # rename all the files 
            os.rename(src, dst) 

 # Driver Code 
if __name__ == '__main__': 
      
    # Calling main() function 
    main() 