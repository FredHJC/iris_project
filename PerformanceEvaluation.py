#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np

def get_crr(class_lab1,class_lab2,class_lab3,target_lab):
    crr_1 = sum(target_lab==class_lab1)/len(target_lab) * 100
    crr_2 = sum(target_lab==class_lab2)/len(target_lab) * 100
    crr_3 = sum(target_lab==class_lab3)/len(target_lab) * 100
    return crr_1,crr_2,crr_3

def get_table(crr_rates, ori_crr_rates):
    print('\n\n')
    dict={'Similarity Measure':['L1 distance measure','L2 distance measure','Cosine similarity measure'],
          'Original Feature':[ori_crr_rates[0],ori_crr_rates[1],ori_crr_rates[2]],
          'Reduced Feature':[crr_rates[0],crr_rates[1],crr_rates[2]]}
    table = pd.DataFrame(dict)
    print ("Correct recognition rate (%):")
    return table


def test_match(threshold, x):
    if x['cos_sim']<=threshold:
        return 1
    else:
        return 0

def ROC(threshold, df):
    df['match'] = df.apply(lambda x: test_match(threshold, x), axis=1)
    n = df.shape[0]
    positive = df[df['idx']==df['class_cos']].shape[0]
    negative = n - positive
    tp = df[(df['match']==1)&(df['idx']==df['class_cos'])].shape[0]
    fp = df[(df['match']==1)&(df['idx']!=df['class_cos'])].shape[0]
    fn = df[(df['match']==0)&(df['idx']==df['class_cos'])].shape[0]

    fmr = fp/negative
    fnmr = fn/positive

    return fmr, fnmr

def PerformanceEvaluation(df_train, df_test, df_test_origin, df_dic):
    df_result = df_dic[100]
    class1, class2, cos_sim, class_cos = df_result['class_1'].values, df_result['class_2'].values, df_result['cos_sim'].values, df_result['class_cos'].values
    class1_origin, class2_origin, cos_sim_origin, class_cos_origin = df_test_origin['class_1'].values, df_test_origin['class_2'].values, df_test_origin['cos_sim'].values, df_test_origin['class_cos'].values

    L1_crr = []
    L2_crr = []
    cos_crr = []

    n_component = range(40,101,20)

    for n in n_component:
        df_tmp = df_dic[n]
        crr_1,crr_2,crr_3 = get_crr(df_tmp['class_1'],df_tmp['class_2'],df_tmp['class_cos'],df_tmp['idx'])
        L1_crr.append(crr_1)
        L2_crr.append(crr_2)
        cos_crr.append(crr_3)

    plt.plot(n_component,L1_crr,label='L1')
    plt.plot(n_component,L2_crr,label='L2')
    plt.plot(n_component,cos_crr,label='Cosine')

    plt.legend(loc = 'upper left')
    plt.xlabel("Dimensionality of the Feature Vector")
    plt.ylabel('Correct Recognition Rate')
    plt.title('Dimensionality vs. CRR')
    plt.show()

    crr_rates = list(get_crr(class1,class2,class_cos,df_result['idx']))
    ori_crr_rates = list(get_crr(class1_origin,class2_origin,class_cos_origin,df_test_origin['idx']))
    print(get_table(crr_rates, ori_crr_rates))

    print('\n\n')
    print ("ROC curve")

    fmr_res = []
    fnmr_res = []
    for threshold in np.linspace(0,1,11):
        fmr, fnmr = ROC(threshold, df_result)
        fmr_res.append(fmr)
        fnmr_res.append(fnmr)

    plt.xlabel("False Match")
    plt.ylabel('False Non-match')
    plt.plot(fmr_res, fnmr_res)
    plt.show()

    print('\n\n')
    table = pd.DataFrame(list(zip(np.linspace(0,1,10), fmr_res, fnmr_res)), columns=['threshold', 'False Match', 'False Non-match'])
    print ("False Match and False Nonmatch Rates with Different Threshold Values")
    print(table)
