
import cv2
import os
import pickle
import random
import numpy as np
import shutil
import csv
from PIL import Image
from skimage import io
import tarfile



source = os.getcwd() + "/test_Images"
X = []
Y = []
files = os.listdir(source)
print("Lenght of Total Images",len(files))

for f in range(0,len(files)):
    print(os.path.basename(files[f]))
    if os.path.basename(files[f]) != ".DS_Store":
        img = cv2.imread(os.path.join(source,os.path.basename(files[f])))
        img_resized = cv2.resize(img, (150, 150))
        X.append(img_resized)
        Y.append(os.path.basename(files[f]))


dataNumpy = np.array(X)
print(len(dataNumpy))
print(np.shape(dataNumpy))

pickle_out = open("XTest.pickle", "wb")
pickle.dump(dataNumpy, pickle_out)
pickle_out.close()
pickle_out = open("YTest.pickle", "wb")
pickle.dump(Y, pickle_out)
pickle_out.close()
