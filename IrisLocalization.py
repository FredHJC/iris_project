
# coding: utf-8

import cv2
import numpy as np
import glob
import math
from scipy.spatial import distance
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

def IrisLocalization(images):
    # Convert all images to grayscale first
    images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
    # Create empty lists to store boundary and center
    boundary = []
    center = []
        
    
    for img in images:
        # Blurring the image
        blurred = cv2.bilateralFilter(img,9,80,80)
        img = blurred


        # Estimating the center of the image
        horizontal_mean = np.mean(img,axis = 0)
        vertical_mean = np.mean(img,axis = 1)
        # The indices with the min values
        X_p = horizontal_mean.argmin()
        Y_p = vertical_mean.argmin()

        # Create a 120 * 120 region centered at the (X_p,Y_p)
        circle_x = img[X_p-60:X_p+60]
        circle_y = img[Y_p-60:Y_p+60]

        # Update the center and estimate center of the pupil
        horizontal_mean = np.mean(circle_y,axis = 0)
        vertical_mean = np.mean(circle_x,axis = 0)

        # Center of the 120*120 region
        xp = horizontal_mean.argmin()
        yp = vertical_mean.argmin()
        # Define center of the pupil
        pupil_center = (xp,yp)

        copy_img = img.copy()
        # Locate the pupil center and show it on the image with one point
        cv2.circle(copy_img,(xp,yp),1,(255,0,0),2)

        # Setting threshold
        masked_img = cv2.inRange(img,0,70)
        result = cv2.bitwise_and(img,masked_img)
        # Apply Canny detector on masked image
        img_edge = cv2.Canny(result, 100, 200)

        # Apply Hough transform to the edged image
        circle = cv2.HoughCircles(img_edge, cv2.HOUGH_GRADIENT, 10, 100)

        min_dist = math.inf
        for i in range(len(circle[0])):

            c=(circle[0][i][0],circle[0][i][1])
            dist = distance.euclidean(pupil_center, c)
            if dist < min_dist:
                min_dist = dist
                k = circle[0][i]

        img_orig = img.copy()
        # Draw the inner boundary
        cv2.circle(img_orig, (int(k[0]), int(k[1])), int(k[2]), (255, 0, 0), 2)

        pupil = k
        radius_pupil = int(k[2])

        # Draw the outer boundary, with the inner boundary adding about 55-60 depending on different people  
        cv2.circle(img_orig, (int(k[0]), int(k[1])), radius_pupil+55, (255, 0, 0), 2)

        # plt.imshow(img_orig,cmap='gray')
        boundary.append(img_orig)
        center.append([int(k[0]),int(k[1]),int(k[2])])
    return boundary,center

