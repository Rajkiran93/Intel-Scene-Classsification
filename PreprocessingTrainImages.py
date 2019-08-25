
import cv2
import os
import pickle
import random
import numpy as np
import shutil
import csv
from PIL import Image
from skimage import io


source = os.getcwd() + "/train_Images"
print(source)
files = os.listdir(source)
print("Lenght of Total Images",len(files))

data = list(csv.reader(open("train.csv")))
X = []
Y = []
X_train = []
test_Image_Names = []   # images
test_Image_ClassName = []   # labels

for f in data:
    test_Image_Names.append(f[0])
    test_Image_ClassName.append(f[1])
test_Image_Names.remove("image_name")
test_Image_ClassName.remove("label")
print("Lenght of test images",len(test_Image_Names))




for f in range(0,len(test_Image_Names)):
    if f != '.DS_Store':
        print(test_Image_Names[f])
        img = cv2.imread(os.path.join(source,test_Image_Names[f]))
        img_resized = cv2.resize(img, (150, 150))
        X.append(img_resized)
        Y.append(test_Image_ClassName[f])


dataNumpy = np.array(X)
print(len(dataNumpy))
print(np.shape(dataNumpy))

pickle_out = open("pickle/X.pickle", "wb")
pickle.dump(dataNumpy, pickle_out)
pickle_out.close()
pickle_out = open("pickle/Y.pickle", "wb")
pickle.dump(Y, pickle_out)
pickle_out.close()

import tarfile
tar = tarfile.open("pickle.tar", "w:tar")
for name in ["X.pickle", "Y.pickle"]:
    tar.add("pickle/"+name)
tar.close()
