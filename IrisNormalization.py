
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


def getRotation(image, degrees):
    # getRotation(image, degree):
    """This function takes normalized image and rotate the rectangle image to specified degree
    :image: input image
    :degree: rotate the rectangle image to specified degree
    :return: rotated image
    """
    res = []
    for degree in degrees:
      pixels = abs(int(512*degree/360))
      if degree == 0:
        res.append(image)
      elif degree > 0:
          res.append(np.hstack([image[:,pixels:],image[:,:pixels]]))
      else:
          res.append(np.hstack([image[:,(512 - pixels):],image[:,:(512 - pixels)]]))

    return res

def IrisNormalization(boundary,centers):
    target = [img for img in boundary]
    normalized=[]
    cent=0
    for img in target:
        #load pupil centers and radius of inner circles
        center_x = centers[cent][0]
        center_y = centers[cent][1]
        radius_pupil=int(centers[cent][2])
        
        iris_radius = 53 # width of space between inner and outer boundary
    
        #define equally spaced interval to iterate over
        nsamples = 360
        samples = np.linspace(0,2*np.pi, nsamples)[:-1]
        polar = np.zeros((iris_radius, nsamples))
        for r in range(iris_radius):
            for theta in samples:
                #get x and y for values on inner boundary
                x = (r+radius_pupil)*np.cos(theta)+center_x
                y = (r+radius_pupil)*np.sin(theta)+center_y
                x=int(x)
                y=int(y)
                try:
                #convert coordinates
                    polar[r][int((theta*nsamples)/(2*np.pi))] = img[y][x]
                except IndexError: #ignores values which lie out of bounds
                    pass
                continue
        res = cv2.resize(polar,(512,64))
        normalized.append(res)
        cent+=1
    return normalized #returns a list of 64x512 normalized images