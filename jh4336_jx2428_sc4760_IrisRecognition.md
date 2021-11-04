# IrisRecognition
This is the main function

### Workflow:
1. Read in the images from the relative path and split them into a training group and a testing group.
2. Looped through every image in the groups. First localize the iris by IrisLocalization function, then normalize and enhance the image by IrisNormalization and IrisEnhancement functions respectively. For each enhanced image, extract the features and store the features and class lables of each image in the list.
3. Performed iris matching in different dimensions under three distances metrics (L1, L2 and cosine similarity)
4. Produced recognition results table using different similarity measures for reduced feature set.
5. Calculated false match rate and false non-match rate based on different threshold
6. Plotted Table 4, False Match and False Non-match Rates with different Threshold Values, and the ROC curve.

## IrisLocalization
The function IrisLocalization(images) preprocesses all the input images and apply Hough transform to detect the pupil circle(inner boundary) and the iris circle(outer boundary) of each image.

### Input
images: A list of train/test images from the CASIA image database.

### Return
boundary: A list of all images with the outer boundary drawn.
center: A list of all pupil centers of each image. Each pupil center is an element of length 3 of the list center. Each element of the list contains x-coordinate, y-coordinate, and the radius of the pupil center.

### Logic
1. Apply bilateral filter to blur each image with the purpose to reduce noise.
2. Project each image both horizontally and vertically and find the local minimum, which is notated by (X_p,Y_p) in the function. This is the approximated pupil center because the local minimum is the darkest region.
3. Create a 120 x 120 region around (X_p,Y_p) as mentioned in Ma's paper; and then recalculate the pupil center using same procedures mentioned in step #2.
4. Apply Canny detector on the masked images and obtain edges around the pupil.
5. Apply Hough transform to the edged image, which detects all possible circles in the image. Then loop through every circle and identify the one with the smallest distance to the approximated center found in step #3. The result is the pupil boundary.
6. Depending on different people, we notice that the iris radius is around 55-60 units larger than the pupil radius. We choose 55 + pupil_radius as the iris radius, assuming that both the pupil and the iris share the same center.
7. Draw the outer boundary and append each image's boundary and center information to desired variables.


## IrisNormalization
The function IrisNromalization(data) counterclockwisely unwrapped the iris area into 64*512 rectangle image.

### Input
boundary: output from IrisLocalization, a list of all images with the outer boundary drawn
center: output from IrisLocalization, a list of all pupil centers of each image.

### Return
normalized: A list of 64x512 normalized images

### Logic
1. Initialized an empty list to store the normalized images;
2. Converted the polar coordinates in the iris region to cartesian coordinates based on the direction of theta.
3. Appended Iris pixel from original image to the new-created normalized list.
4. Kept original image coordinates within the fixed boundary  i.e(280,320), discarded the rest.
5. Resized input image to a rectangular 64x512 sized image.


## IrisEnhancement

IrisEnhancement(normalized) divided the normalized image into thirty three 32x32 smaller regions and enhanced each region using Histogram Equalization.

### Input:
normalized: normalized image, output from IrisNormalization.

### Return:
Enhanced images in a list of array

### Logic
1. Divided the normalized image into 32x32 regions;
2. Performed histogram equalization for each divided region;
3. Mapped every enhanced 32x32 region back to the original normalized image.


## getRotation

getRotation(image, degree) will rotate the enhanced normalized image to specified degree

### input:
image: enhanced normalized image
degree: specified rotation degree in a list

### Return:
Image after rotation

### Logic
1. Obtained image pixels based on rotation degree;
2. Considered three cases depending on whether rotation angle is negative, 0 or positive;
3. Appended corresponding rotation pixels to result list from the enhanced normalized image.

## FeatureExtraction

### Input
enhanced: the function inputs all the enhanced images from IrisEnhancement.

### Return
feature_vec: A numpy array with each element containing 1536 feature components aas defined in Li Ma's paper.

### Functions
M(x,y,f): inputs x: x-coordinate, y: y-coordinate, and f: frequency and calculates the M1 modulating function mentioned in paper.

Gabor(x,y,dx,dy,f): inputs x,y, dx: space constant x, dy: space constant y, and f. It performs the gabor filterd defined in paper.

block(dx,dy,f): inputs dx,dy, and f. Implements gabor filtering on an 8 x 8 region with predetermined parameters dx,dy, and f.

get_vector(vector1,vector2): inputs two filtered images and computes the mean and standard deviation of each 8 x 8 block. Finally it returns a featured vector with means and stds of two different channels appended in the specified order as mentioned in Li Ma's paper.

### Logic
1. Define two channels using specified space constants as mentioned in Li Ma's paper.
2. Obtain the ROI by slicing the normalized image to 48 x 512, as defined in paper.
3. Implement the convolution method on two channels and obtain two filtered images.
4. Run get_vector() function using the two filtered images from step #3
5. Obtain the desired feature vector.

## IrisMatching
### Input
It inputs trained feature vector and test feature vector
### Return
Returns dataframes and crr arrays where are crr results from different distance measures and number of features.

### Functions
chunks(lst,n): Yield successive n-sized chunks from lst.
compute_dis(test_feature, target_feature, norm): Compute the distance measure between test feature and target feature. L1 and L2 distance measures are represented by norm = 1 or 2 and other values of norm indicate that cosine similarity is calculated.
get_class(p,df_train, label): Get the class labels for each distance measure by calling compute_dis() function. Return a dataframe with each class labels concatenated to the input dataframe.
Feature_num(df_train, df_test, n_component): input the train and test dataframe, and a list of feature numbers. Set df_dic as an empty dictionary. Each feature number is stored in df_dic as key. Then LDA is performed so that the original feature is transformed to the reduced feature with specified feature number. Then the entire test dataframe is stored to df_dic as the value. It returns df_dic at the end.

### Logic
1. Create the trained dataframe where each image is rotated at 7 angles as mentioned in Li Ma’s paper. In the dataframe, idx indicates the actual class number; img_idx indicates which train image it belongs in the class; degree indicates what angle the image is rotated.
2. Similarly, create the test dataframe where for each class there are four testing images and no rotations are performed.
3. Choose a series of feature numbers we want to test on and call the Feature_num() function.

## PerformanceEvaluation
PerformanceEvaluation(df_train, df_test, df_test_origin, crrs, df_result) will input both train and test data frames, original feature dataframe and dictionary that stores feature numbers in LDA. The function will output correct recognition rate table and corresponding ROC curve. Performance is evaluated by three different distance measures (L1, L2 and cosine similarity) and dimensionality reduction in LDA.

### SubFunctions
1. get_crr(class_lab1,class_lab2,class_lab3,target_lab): input classification labels returned from IrisMatching, and returned crr_1,crr_2,crr_3, which are Correct Recognition Rate for class L1, L2 and cosine similarity.
2. get_table(crr_rates): input crr_rates for 3 classes obtained from function get_crr() and returned a table form of correct recognition rate for class L1, L2 and cosine similarity .
3. test_match(threshold, x): input predetermined threshold values and bound variable x for lambda function and returned 1 (accepted) if it was within the threshold, otherwise returned 0 (rejected).
4. ROC(threshold, df): input predetermined threshold and test data frame, and output false positive rate and true positive rate used to draw ROC graph.

## Peer Evaluation

| Part      | Team member     | 
| ---------- | :-----------:  | 
| IrisLocalization     | Jiashu Xia  |
| IrisNormalization     | Siqi Chen, Jiachen Huang |
| Iris Enhancement     | Siqi Chen   |
| FeatureExtraction     | Jiashu Xia     |
| IrisMatching     | Jiachen Huang, Jiashu Xia    |
| PerformanceEvaluation     | Jiachen Huang    |
| IrisRecognition     | Jiachen Huang     |



