# -*- coding: utf-8 -*-
"""
File name: hog_svm.py
Created on Tue May  4 11:20:16 2021

@author: Olaniyi Taofeeq O
"""


import numpy as np
import pandas as pd
import cv2
import os
import sys
import math
import time

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
#from sklearn.externals 
import joblib
from skimage.feature import hog
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# dictionary of labels


def datagen(img_folder):
    """
    Function: datagen 
    
    Input: 
        List of filenames with their absolute paths
    
    Output: Train data and labels depending on mode value
    
    Description: This function computes HOG features for each image in the data/image/train folder, assigns label to the descriptor vector of the image and returns the final train/test data and labels matrices used for feeding the SVM in training phase or predicting the label of test data.
    
    """

    data = []
    label = []


    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
            filename = os.path.join(img_folder, dir1,  file)

            # read image
            img = cv2.imread(filename)
            #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            
            #img = cv2.medianBlur(img,3)
            #image = Standardizer.read_image(filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img,(200,200))
            image = cv2.medianBlur(img, 3)

            # threshold
            # Set total number of bins in the histogram
            bins_num = 256



            # Get the image histogram
            hist, bin_edges = np.histogram(image, bins=bins_num) 

            # Get normalized histogram if it is required
           
            hist = np.divide(hist.ravel(), hist.max())

            # Calculate centers of bins
            bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.

            # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
            weight1 = np.cumsum(hist)
            weight2 = np.cumsum(hist[::-1])[::-1]

            # Get the class means mu0(t)
            mean1 = np.cumsum(hist * bin_mids) / weight1
            # Get the class means mu1(t)
            mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]

            inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

            # Maximize the inter_class_variance function val
            index_of_max_val = np.argmax(inter_class_variance)

            threshold1 = bin_mids[:-1][index_of_max_val]
          # print("Otsu's algorithm implementation thresholding result: ", threshold1)
          # print(image)


            thresh = cv2.threshold(image, threshold1, 255, cv2.THRESH_BINARY)[1]

            #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            #dilate = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)

    # get absolute difference between dilate and thresh
            #diff = cv2.absdiff(dilate, thresh)

        # invert
            #edges = 255 - diff
            # compute HOG features
            
            
            
            
            # = FeatureExtractor()
            #otsu_threshold, image_result = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,)
            #edges=cv2.Canny(image_result, 200, 300)
            #hog_feature, spatial, color, canny = featureExtractor.extract_features(image)
            hog_feature, hog_image = hog(thresh, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm= 'L2',visualize=True)
            #print((len(hog_feature))
            #image_path = os.path.join(file, filename)
            image_label = os.path.splitext(os.path.basename(dir1))[0]
#            image_info = {}
#            image_info['image_path'] = image_path
#            image_info['image_label'] = image_label
            #images_labels_list.append(image_info)
            #print(image_label)
    
            # append descriptor and label to train/test data, labels
            data.append(hog_feature)
            label.append(image_label) 
           

# return data and label
    return data, label

def main():
    
    # list of training and test files
    train_folder = r"Alphabets1/"
    #train_folder = r"data/images/Asl_alphabet_train_250/"
    #train_folder = r"data/images/Numbers/"
    #train_folder = r"data/images/Alphabets/"
    #train_folder = r"data/images/Alphabets1/"
    #training data & labels
    data, label = datagen(train_folder)
  
    print(" Constructing training/testing split...")
    (trainData, testData, trainLabels, testLabels) = train_test_split(
	np.array(data), label, test_size=0.33, random_state=10)
    
    #scaler = StandardScaler()
    
    #scaler.fit(trainData)
    #trainData = scaler.transform(trainData)
    #testData = scaler.transform(testData)
    # training phase: SVM , fit model to training data ------------------------------
    model = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    #model = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')
    model.fit(trainData, trainLabels)
    # predict labels for test data
    predictions = model.predict(testData)
    
    # compute accuracy
    accuracy = accuracy_score(testLabels, predictions) * 100
    
    print (" Mahmud et al 2018 Implementation" )
    print (" HOG parameters are  size = 128 x 128 pixel, Orientations=9, Pixels_per_cell=(8, 8), Cells_per_block=(2, 2) at K=1" )
    print("\nAverage Accuracy with KNN: %.2f" % accuracy + "%")
    
#c_names = [name[13:] for name in glob.glob('./data/images/train/*')]
    cm1 = confusion_matrix(predictions, testLabels)
    plt.figure(figsize=(10,10))
    sns.heatmap(cm1, annot=True)
    #cf_matrix = confusion_matrix(testLabels, predictions)
    #print(cf_matrix)
    #plot_confusion_matrix(cm1,normalize= True)
    plt.show()
    
    # classification report for precision, recall f1-score and accuracy
    matrix = classification_report(predictions, testLabels)
    print('Classification report : \n',matrix)
    
    
    # Save the model:
    # Save the Model
    joblib.dump(model, 'knn_model.pkl')# -*- coding: utf-8 -*-

if __name__ == "__main__": 
    start_time = time.time()
    main()
    print('Execution time: %.2f' % (time.time() - start_time) + ' seconds\n')