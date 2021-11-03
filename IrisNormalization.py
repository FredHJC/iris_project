
# coding: utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
import copy

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

def IrisNormalization(boundary,centers, rotation_flag):
    def apply_transform(iris_width, theta_range, x_p, y_p, pupil_radius, polar, img):
        for Y in range(iris_width):
            for theta in theta_range:
                # get x and y for values on inner boundary
                # iris_width + pupil_radius = outer boundary radius
                outer_radius = Y + pupil_radius

                x = int(x_p + (outer_radius) * np.cos(theta))
                y = int(y_p + (outer_radius) * np.sin(theta))


                # theta = 2*pi*X => X = theta/(2*pi)
                # Here theta is only one slice angle, need to multiply by 360
                X  = int((360 * theta)/(2 * np.pi))

                try:
                    # convert polar coordinates to cartesian coordinates
                    polar[Y][X] = img[y][x]

                # discard values out of boundary
                except IndexError:
                    pass
                continue

            # reshape image into size 64*512 size
        polar_resize = cv2.resize(polar,(512,64))
        return polar_resize


    # x,y location for pupil and pupil radius

    # x_p = centers[center][0]
    # y_p = centers[center][1]
    # pupil_radius = int(centers[center][2])

    # iris width, the distance between inner boundary and outer boundary
    iris_width = 55

    # define equally spaced interval to iterate over
    theta_range = np.linspace(0,2*np.pi,360)[:-1]

    # create an empty list of array for polar coordinates
    # for normalized location(X,Y), X is the 360 sliced circumstance
    # Y is the iris width
    polar = np.zeros((iris_width,360))

    tmp = list(zip(boundary,centers))

    normalized = list(map(lambda x: apply_transform(iris_width, theta_range, x[1][0], x[1][1], x[1][2], copy.deepcopy(polar), x[0]), tmp))

    if not rotation_flag:
        return normalized
    else:
        return sum(list(map(lambda p: getRotation(p, [-9,-6,-3,0,3,6,9]), normalized)), [])
