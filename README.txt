Environment used:-
1)Google Colab was been used for the training purpose.

Steps To Build and predict:-
1)All the training images should be in "train_Images" folder.
2)All the testing images should be in "test_Images" folder.
3)Open the training files "VGG16.ipynb" and "VGG19.ipynb" on collar.
4)Run the script "PreprocessingTrainImages.py" this will generate a "pickle.tar" file.
5)"picle.tar" should be uploaded on to the Google drive and click on the share which gives you an shareable link, get the "id" from this link paste in both "VGG16.ipynb" and "VGG19.ipynb" files and start running the files.
6)Once the training is done the script generates "VGG16.h5" and "VGG19.h5" models, we need to download these model files and move to "Final Submission" file.
7)Now run "PreprocessingTestImages.py" script this will generate "XTest.pickle" and "YTest.pickle" files which is the test data.
8)Finally run script "predict" this will give "test.csv" file.