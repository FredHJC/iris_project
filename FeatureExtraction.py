import cv2
import numpy as np
import glob
import math
import scipy
from scipy.spatial import distance
from scipy import signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics


#modulating function as defined in paper
def M(x ,y, f):
    res = np.cos(2 * np.pi * f * math.sqrt(x ** 2 + y ** 2))
    return res

# Define the Spatial filter
def Gabor(x, y, dx, dy, f):
    # print(x, y, dx, dy, f)
    gabor = (1/(2 * np.pi * dx * dy)) * np.exp(-1/2 * (x ** 2/(dx ** 2) + y ** 2/(dy ** 2))) * M(x,y,f)
    return gabor


# Apply the defined filter to an 8 by 8 block 
def block(dx, dy, f):
    feature = np.zeros((8,8))
    for i in range(8):
        for j in range(8):
            feature[i,j] = Gabor((-4+j),(-4+i),dx,dy,f)
    return feature


# The function inputs the two filtered image, calculates mean and std for each block of each channel, 
# and returns the desired vector V with all means and stds appended to it.
def get_vector(vector1,vector2):
    feature_vec = []
    # ranges are determined by 48/8 = 6 and 512/8 = 64
    for i in range(6):
        for j in range(64):
            # Loop over each 8 by 8 block to get the feature
            x1 = 8 * i
            x2 = x1 + 8
            y1 = 8 * j
            y2 = y1 + 8
            
            # Filtered image block
            c1 = vector1[x1:x2,y1:y2]
            c2 = vector2[x1:x2,y1:y2]
            
            c1 = np.abs(c1)
            c2 = np.abs(c2)
            
            # Follow the calculation steps in Ma's paper
            # Channel 1 mean and standard deviation
            m1 = np.mean(c1)
            sigma1 = np.mean(np.abs(c1-m1))
            feature_vec.append(m1)
            feature_vec.append(sigma1)
            
            # Channel 2 mean and standard deviation
            m2 = np.mean(c2)
            sigma2 = np.mean(np.abs(c2-m2))
            feature_vec.append(m2)
            feature_vec.append(sigma2)
            
    return feature_vec

# Inputs a single normalized image
def FeatureExtraction(enhanced_normal):
    f = 2/3
    # Get two channels using the parameters defined in paper
    channel1 = block(3, 1.5, f)
    channel2 = block(4, 1.5, f)
    
    feature_vec = []
    

    # enhanced_normal has length 64 and enhance_normal has length 512
    # Define a 48 by 512 region as ROI
    ROI = enhanced_normal[:48,:]

    filtered1 = scipy.signal.convolve2d(ROI,channel1,mode='same')
    filtered2 = scipy.signal.convolve2d(ROI,channel2,mode='same')
    
    vector = get_vector(filtered1,filtered2)
    feature_vec.append(vector)
    # len(feature_vec) == 1536
    return np.array(feature_vec).flatten()