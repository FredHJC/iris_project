import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def compute_dis(test_feature, target_feature, norm):
    """Compute different norm and cos similarity"""
    if norm == 1:
        return np.sum(np.abs(test_feature-target_feature))
    elif norm == 2:
        return np.sum(np.square(test_feature-target_feature))
    else:
        return 1-np.dot(test_feature,target_feature.T)/(np.linalg.norm(test_feature)*np.linalg.norm(target_feature))


def get_class(p, df_train, label):
    """Compute the matched class, as well as the similarity score"""
    train_templates = df_train[label].values
    dis_1 = list(map(lambda x: compute_dis(p[label], x, 1),train_templates))
    dis_2 = list(map(lambda x: compute_dis(p[label], x, 2),train_templates))
    dis_cos = list(map(lambda x: compute_dis(p[label], x, 3),train_templates))
    min_idx_1 = np.argmin(dis_1)
    min_idx_2 = np.argmin(dis_2)
    min_idx_cos = np.argmin(dis_cos)

    # assign result to the new df row
    p['class_1'] = list(df_train['idx'].values)[min_idx_1]
    p['class_2'] = list(df_train['idx'].values)[min_idx_2]
    p['cos_sim'] = min(dis_cos)
    p['class_cos'] = list(df_train['idx'].values)[min_idx_cos]
    return p

def feature_num(df_train, df_test, n_component):
    """Input a test dataframe and a list of feature numbers to implement LDA"""

    clf = LDA(n_components=n_component)
    clf.fit(list(df_train['feature'].values), list(df_train['idx'].values))
    df_train['reduced_feature']=df_train['feature'].apply(lambda x: clf.transform(np.array(x).reshape(1, -1)))
    df_test['reduced_feature']=df_test.copy()['feature'].apply(lambda x: clf.transform(np.array(x).reshape(1, -1)))

    # Class labels using L1, L2, and Cosine Similarity measures
    df_test = df_test.apply(lambda x: get_class(x, df_train.copy(), 'reduced_feature'), axis=1)

    return df_test

def IrisMatching(feature_vector_train,feature_vector_test):
    # slice the train vectors
    sliced_res_train = list(chunks(feature_vector_train,21))
    list_of_dataframes_train = []

    for i, eyes in enumerate(sliced_res_train):
        # i from 0 to 107
        sliced_eyes = list(chunks(eyes,7))
        for j, eye in enumerate(sliced_eyes):
            # j from 0 to 3
            degree = [-9,-6,-3,0,3,6,9]
            img_idx = [j]*7
            idx = [i]*7
            # transform vectors to dataframe
            df_tmp = pd.DataFrame({'idx':idx, 'img_idx':img_idx, 'degree':degree, 'feature': eye })
            list_of_dataframes_train.append(df_tmp)
    df_train = pd.concat(list_of_dataframes_train)

    sliced_res_test = list(chunks(feature_vector_test,4))

    list_of_dataframes_test = []
    for i, eyes in enumerate(sliced_res_test):
        # i from 0 to 107
        for j, eye in enumerate(eyes):
            # j from 0 to 3
            img_idx = [j]
            idx = [i]
            df_tmp = pd.DataFrame({'idx':idx, 'img_idx':img_idx, 'feature':[np.array(eye)]})
            list_of_dataframes_test.append(df_tmp)

    df_test = pd.concat(list_of_dataframes_test)

    df_test['feature']=df_test['feature'].apply(lambda x: np.array(x).flatten())
    df_train['feature']=df_train['feature'].apply(lambda x: np.array(x).flatten())

    n_component = range(40,101,20)
    df_dic = {}

    for i,n in enumerate(n_component):
        df_tmp = feature_num(df_train.copy(),df_test.copy(),n)
        df_dic[n] = df_tmp
    df_test_origin = df_test.apply(lambda x: get_class(x, df_train.copy(), 'feature'), axis=1)

    return df_train, df_test, df_test_origin, df_dic