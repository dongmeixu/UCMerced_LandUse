# -*- coding: utf-8 -*-
import argparse as ap
import cv2
import numpy as np
import os
import joblib
from scipy.cluster.vq import *
from sklearn import preprocessing
import math
import sys
import numpy
from sklearn import metrics
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import codecs
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import cross_validation

''''' 
使用词袋模型对图像进行分类： 
1、数据格式，文件夹名即类别，每个文件夹下是一类图像 
2、提取sift特征，将所有图像的sift特征放在一起，进行聚类，聚出n个视觉词 
3、计算每幅图像有哪些视觉词，统计出词频矩阵 
4、根据矩阵计算idf，进而得到tfidf矩阵，并对其进行L2归一化（向量中每个元素除以向量的L2范数->x/平方和开根号） 
5、使用一般分类模型对其进行分类，计算P,R,F 
'''

"""
    import cv2
ImportError: DLL load failed: %1 ������Ч�� Win32 Ӧ�ó���
python
"""
def load_data(path):
    '''''
    每个文件夹下是一种图片
    :param path:种类文件夹路径
    :return: 图片路径列表和标签列表
    '''
    categories = os.listdir(path)
    img_pathes = []
    labels = []
    for path, dirs, files in os.walk(path):
        img_pathes.extend([os.path.join(path, file) for file in files])
        print path
        if len(files) > 0:
            labels.extend([path.split('/')[-1]] * len(files))
            # print len(img_pathes),img_pathes
    # print len(labels),labels
    return img_pathes, labels


def cal_bow(image_paths, numWords):
    '''''
    使用bag of word方法提取图像特征
    :param image_paths:
    :return:
    '''
    # numWords = 100
    # 关键点检测对象
    global descriptor
    fea_det = cv2.FeatureDetector_create("SIFT")
    # 特征提取对象
    des_ext = cv2.DescriptorExtractor_create("SIFT")

    # List where all the descriptors are stored
    des_list = []

    for i, image_path in enumerate(image_paths):
        im = cv2.imread(image_path)
        # print "Extract SIFT of %s image, %d of %d images" % (image_paths[i], i, len(image_paths))
        kpts = fea_det.detect(im)
        # 可能存在没有检测出特征点的情况
        # print len(kpts)
        # des有k行m列，每行代表一个特征，m是固定的特征维数
        kpts, des = des_ext.compute(im, kpts)
        # print(len(kpts), des)
        des_list.append((image_path, des))

    # Stack all the descriptors vertically in a numpy array
    # image_path为图片路径，descriptor为对应图片的特征
    # 将所有特征纵向堆叠起来,每行当做一个特征词
    descriptors = des_list[0][1]
    for image_path, descriptor in des_list[1:]:
        # vstack对矩阵进行拼接，将所有的特征word拼接到一起
        if descriptor is not None:
            # print descriptor.shape, descriptors.shape
            descriptors = np.vstack((descriptors, descriptor))
        # 对特征词使用k-menas算法进行聚类
    print(descriptors)
    print(descriptors.shape)  # (193470L, 128L)
    print(descriptors.dtype)  # float32
    print "Start k-means: %d words, %d key points" % (numWords, descriptors.shape[0])
    # 最后输出的结果其实是两维的,第一维是聚类中心,第二维是损失distortion
    voc, variance = kmeans(descriptors, numWords, iter=1)
    # print(voc, variance)
    # print(voc.shape, voc.max(), voc.min(), voc.mean())   # ((500L, 128L), 196.7922, 0.07435898, 26.891306)

    # 初始化一个bag of word矩阵，每行表示一副图像，每列表示一个视觉词，下面统计每副图像中视觉词的个数
    print(len(image_paths))  # 400
    print(len(des_list)) # 400
    im_features = np.zeros((len(image_paths), numWords), dtype="float32")
    for i in range(len(image_paths)):
        # 计算每副图片的所有特征向量和voc中每个特征word的距离，返回为匹配上的word

        if des_list[i][1] is not None:
            # 根据聚类中心将所有数据进行分类des_list[i][1]为数据, voc则是kmeans产生的聚类中心.
            # vq输出有两个:一是各个数据属于哪一类的label,二是distortion

            print(type(des_list[i][1]))
            # print(type(voc))
            words, distance = vq(des_list[i][1], voc)

        for w in words:
            im_features[i][w] += 1

            # Perform Tf-Idf vectorization
    nbr_occurences = np.sum((im_features > 0) * 1, axis=0)
    idf = np.array(np.log((1.0 * len(image_paths) + 1) / (1.0 * nbr_occurences + 1)), 'float32')

    # L2归一化
    im_features = im_features * idf
    im_features = preprocessing.normalize(im_features, norm='l2')
    return im_features


def train_clf2(train_data, train_tags):
    # clf = SVC(kernel = 'linear')#default with 'rbf'
    clf = LinearSVC(C=1100.0)  # default with 'rbf'
    clf.fit(train_data, train_tags)
    return clf


def evaluate(actual, pred):
    """
     ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.

    :param actual:
    :param pred:
    :return:
    """
    m_precision = metrics.precision_score(actual, pred, average="macro")
    m_recall = metrics.recall_score(actual, pred, average="macro")
    print 'precision:{0:.3f}'.format(m_precision)
    print 'recall:{0:0.3f}'.format(m_recall)
    print 'f1-score:{0:.8f}'.format(metrics.f1_score(actual, pred, average="macro"));


''''' 
'''
# 提取图片特征并保存 图像大小为20*20
path = 'Images/'
img_pathes, labels = load_data(path)
im_features = cal_bow(img_pathes, numWords=700)
joblib.dump((im_features, labels), "bof_700.pkl", compress=3)

# 训练并测试
im_features, labels = joblib.load("bof_700.pkl")
print(im_features.shape, labels)
X_train, X_test, y_train, y_test = \
    cross_validation.train_test_split(im_features, labels, test_size=0.3, random_state=0)
clf = train_clf2(X_train, y_train)

pred = clf.predict(X_test)
print pred
print y_test
evaluate(y_test, pred)


#
# test = []
# test.append((1, 'a'))
# test.append((2, 'b'))
# test.append((3, 'v'))
# test.append((4, 'd'))
# test.append((5, 'e'))
# descriptors = test[0][1]
# print descriptors
# print(test[1:])
# for image_path, descriptor in test[1:]:
#     # vstack对矩阵进行拼接，将所有的特征word拼接到一起
#     # print descriptor
#     # if descriptor != None:
#     descriptors = np.vstack((descriptors, descriptor))
# print(descriptors)
