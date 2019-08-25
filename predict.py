import numpy as np
import tensorflow as tf
import random as rn
import os
import cv2
import pickle
import keras
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
import csv


XTEST = []
YTEST = []

def classifyImages():
    XTEST = pickle.load(open(os.getcwd() + "/XTest.pickle", "rb"))
    YTEST = pickle.load(open(os.getcwd() + "/YTest.pickle", "rb"))
    print(XTEST.shape)
    classifierLoadVGG16 = tf.keras.models.load_model('VGG16.h5')
    classifierLoadVGG19 = tf.keras.models.load_model('VGG19.h5')
    classifierLoadVGG16.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    classifierLoadVGG19.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    with open('submission.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(["image_name","label"])
        for i in range(len(XTEST)):
            print(i)
            xxx = XTEST[i].reshape(1, 150, 150, 3)
            resultVGG16 = classifierLoadVGG16.predict(xxx)
            resultVGG19 = classifierLoadVGG19.predict(xxx)
            classProbabVGG16 = (resultVGG16.tolist())[0]
            classProbabVGG19 = (resultVGG19.tolist())[0]
            maxVGG16 = max(classProbabVGG16)
            maxVGG19 = max(classProbabVGG19)
            predictedClass = ""
            if maxVGG16 > maxVGG19:
                predictedClass = str(classProbabVGG16.index(max(classProbabVGG16)))
            else:
                predictedClass = str(classProbabVGG19.index(max(classProbabVGG19)))
            filewriter.writerow([str(YTEST[i]),predictedClass])



if __name__ == '__main__':
    classifyImages()

