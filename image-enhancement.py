import numpy as np
from cv2 import *
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
img_width, img_height = 150, 150 



train_data_dir = 'dataset/train'
validation_data_dir = 'dataset/test'
nb_train_samples = 4353
nb_validation_samples = 1098

img_dir = "dataset/train/ALZ"
img_dir_2 = "dataset/train/NALZ"
img_dir_3 = "dataset/test/ALZ"
img_dir_4 = "dataset/test/NALZ"

'''
data = []
for f1 in files:
    img = cv2.imread(f1)
    data.append(img)
    


print(len(data))
print(data[0].shape)
plt.imshow(data[0]);
'''
kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1, 9,-1],
                              [-1,-1,-1]])

########
def sharpen(files):
	
	for i in files:
		image = cv2.imread(i)
		sharpened = cv2.filter2D(image, -1, kernel_sharpening)
		imwrite(i , sharpened)

	return files

####
print("file extracting")
data_path = os.path.join(img_dir, '*g')
files = glob.glob(data_path)
sharpen(files)
print("train_x1 extracted and sharpened")



data_path_2 = os.path.join(img_dir_2, '*g')
files_2 = glob.glob(data_path_2)
sharpen(files_2)
print("train_x2 extracted and sharpened")




data_path_3 = os.path.join(img_dir_3, '*g')
files_3 = glob.glob(data_path_3)
sharpen(files_3)


print("test_x1 extracted and sharpened")

data_path_4 = os.path.join(img_dir_4, '*g')
files_4 = glob.glob(data_path_4)
sharpen(files_4)
print("test_x2 extracted and sharpened")



##  1 is for non alz and 0 for alz

### train_x here the x is referred to as images and y is referred to as the labels  


# train_y1 = np.ones((1793, 1))         ### Alz train data
# train_y2 = np.zeros((2560, 1))		  ### NonAlz train data
# test_y1 = np.ones((640, 1))		  ### Alz test data
# test_y2 = np.zeros((458, 1))		  ### NonAlz test data	

# print("file extracted")
# print("training started")

# train_x = np.concatenate((train_x1, train_x2), axis=0);
# train_y = np.concatenate((train_y1, train_y2), axis=0);
# print(train_x.shape)
# print(train_y.shape)

# test_x = np.concatenate((test_x1, test_x2), axis=0);
# test_y = np.concatenate((test_y1, test_y2), axis=0);



# print(test_x.shape)
# print(test_y.shape)

