import cv2
import numpy as np
import glob
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import pickle

from IrisLocalization import IrisLocalization
from IrisNormalization import IrisNormalization
from ImageEnhancement import ImageEnhancement
from FeatureExtraction import FeatureExtraction
from IrisMatching import IrisMatching
from PerformanceEvaluation import PerformanceEvaluation

images_train = [cv2.imread(file) for file in sorted(glob.glob('CASIA Iris Image Database (version 1.0)/*/1/*.bmp'))]
#running Localization, Normalization,Enhancement and Feature Extraction on all the training images
boundary_train,centers_train=IrisLocalization(images_train)
normalized_train=IrisNormalization(boundary_train,centers_train, True)
enhanced_train=ImageEnhancement(normalized_train)
feature_vector_train=FeatureExtraction(enhanced_train)
print("Finish training data processing")

images_test = [cv2.imread(file) for file in sorted(glob.glob('CASIA Iris Image Database (version 1.0)/*/2/*.bmp'))]
#running Localization, Normalization,Enhancement and Feature Extraction on all the testing images
boundary_test,centers_test=IrisLocalization(images_test)
normalized_test=IrisNormalization(boundary_test,centers_test, False)
enhanced_test=ImageEnhancement(normalized_test)
feature_vector_test=FeatureExtraction(enhanced_test)
print("Finish testing data processing")

with open('feature_vector_train.pkl', 'wb') as f:
    pickle.dump(feature_vector_train, f)

with open('feature_vector_test.pkl', 'wb') as f:
    pickle.dump(feature_vector_test, f)

feature_vector_train = np.load('feature_vector_train.pkl', allow_pickle=True)
feature_vector_test = np.load('feature_vector_test.pkl', allow_pickle=True)

print("Backup data saved")

df_train, df_test, df_test_origin, df_dic = IrisMatching(feature_vector_train,feature_vector_test)
print("Finish matching")

PerformanceEvaluation(df_train, df_test, df_test_origin, df_dic)
