# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 00:05:08 2021

@author: BELLO
"""


import cv2
import numpy as np
from PIL import Image
from keras import models
import joblib
from skimage.feature import hog

import os
import traceback
import logging
import numpy as np
import cv2
#from sklearn.externals 
import joblib
from skimage.feature import hog

#from common.config import get_config
#from common.image_transformation import resize_image


def get_image_from_label(label):
    testing_images_dir_path = get_config('Camera/')
    image_path = os.path.join(testing_images_dir_path, label,)
    image = cv2.imread(image_path)
    return image

logging_format = '[%(asctime)s||%(name)s||%(levelname)s]::%(message)s'
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"),
                    format=logging_format,
                    datefmt='%Y-%m-%d %H:%M:%S',)
logger = logging.getLogger(__file__)


#Load the saved model
model = joblib.load('knn_model.pkl')
def main():
    cam = cv2.VideoCapture(0)
    while True:
        ret, frame = cam.read()
    #frame = cv2.flip(frame, 1)
        img = cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), thickness=5, lineType=8, shift=0)
        if not ret:
            logger.error("Failed to capture image!")
            continue

        cv2.imshow("Webcam recording", img)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(200,200))
        image = cv2.medianBlur(img, 3)
        

        #Our keras model used a 4D tensor, (images x height x width x channel)
        #So changing dimension 128x128x3 into 1x128x128x3 
        #img_array = np.expand_dims(img_array, axis=0)
        bins_num = 256
#
#

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
        #print("Otsu's algorithm implementation thresholding result: ", threshold1)
      # print(image)


        thresh = cv2.threshold(image, threshold1, 255, cv2.THRESH_BINARY)[1]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        dilate = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)

# get absolute difference between dilate and thresh
        diff = cv2.absdiff(dilate, thresh)

    # invert
        edges = 255 - diff
        try:
            #compute HOG features
            
            
            
            
            # = FeatureExtractor()
            #otsu_threshold, image_result = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,)
            #edges=cv2.Canny(image_result, 200, 300)
            #hog_feature, spatial, color, canny = featureExtractor.extract_features(image)
            hog_feature, hog_image = hog(edges, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm= 'L2',visualize=True)
                

    #Calling the predict method on model to predict 'me' on the image
            predicted_labels = model.predict([hog_feature])
            predicted_label = predicted_labels[0]
            logger.info("Predicted label = {}".format(predicted_label))
#            predicted_image = get_image_from_label(predicted_label)
#            predicted_image = resize_image(predicted_image, 200)
#            cv2.imshow("Prediction = '{}'".format(
#                predicted_label), predicted_image)
            print('Predicted: {} '.format(
                        predicted_label))
        except Exception:
            exception_traceback = traceback.format_exc()
            logger.error("Error applying image transformation")
            logger.debug(exception_traceback)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()
    cv2.destroyAllWindows()
    logger.info("The program completed successfully !!")


if __name__ == '__main__':
    main()
