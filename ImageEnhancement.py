
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

#Equalizes the histogram of the image

def enhance(image):
    enhanced_normal = image.copy()

    # range for row and column
    nrow = int(enhanced_normal.shape[0]/32) # 2
    ncol = int(enhanced_normal.shape[1]/32) # 16

    # The following loop will first define each 32*32 region in normalized image
    # and then perform histogram equalization in each divided region
    
    for row in range(nrow):
        for col in range(ncol):

            # Devide normalized image into 32*32 regions
            region = enhanced_normal[row*32 : (row+1)*32, col*32 : (col+1)*32]

            # Perform histogram equalization for each 32*32 region
            enhanced_normal[row*32 : (row+1)*32, col*32 : (col+1)*32] = cv2.equalizeHist(region)
    return np.array(enhanced_normal)

def ImageEnhancement(normalized):
    enhanced=[]
    for res in normalized:
        res = res.astype(np.uint8)
        im=enhance(res)
        enhanced.append(im)
    return enhanced