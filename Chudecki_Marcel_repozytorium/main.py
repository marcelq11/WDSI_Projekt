

import os
import random
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pandas
import xml.etree.ElementTree as ET
from pathlib import Path

def load_data(path):
    data = []
    for i in os.listdir(path+'annotations/'):
        tree = ET.parse(path+'annotations/'+i)
        filename = tree.find('filename')
        # print(filename.text)
        objects = tree.findall('object')
        # print(len(objects))
        for i in objects:
            x_min = i.find('bndbox/xmin')
            y_min = i.find('bndbox/ymin')
            x_max = i.find('bndbox/xmax')
            y_max = i.find('bndbox/ymax')
            # print(x_min.text,y_min.text,x_max.text,y_max.text)
            image = cv2.imread(path+'images/'+filename.text)
            image_cropped = image[int(y_min.text):int(y_max.text), int(x_min.text):int(x_max.text)]
            if str(i.find('name').text) == 'speedlimit':
                data.append({'image': image_cropped, 'label': str(i.find('name').text)})
            else:
                data.append({'image': image_cropped, 'label': 'other'})
    return data

def learn_bovw(data):
    dict_size = 128
    bow = cv2.BOWKMeansTrainer(dict_size)

    sift = cv2.SIFT_create()
    for sample in data:
        kpts = sift.detect(sample['image'], None)
        kpts, desc = sift.compute(sample['image'], kpts)

        if desc is not None:
            bow.add(desc)

    vocabulary = bow.cluster()

    np.save('voc.npy', vocabulary)

def extract_features(data):
    sift = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher_create()
    bow = cv2.BOWImgDescriptorExtractor(sift, flann)
    vocabulary = np.load('voc.npy')
    bow.setVocabulary(vocabulary)

    for sample in data:
        kpts = sift.detect(sample['image'], None)
        imgDes = bow.compute(sample['image'], kpts)
        if imgDes is not None:
            sample.update({'desc': imgDes})
        else:
            sample.update({'desc': np.zeros((1, 128))})
    return data

def main():
    main = Path('main.py')
    main_path = main.parent.absolute()
    absolute_path=main_path.parent
    data_train = load_data(str(absolute_path) + '/train/')
    data_test = load_data(str(absolute_path) + '/test/')
    print(len(data_train))
    print(len(data_test))
    # learn_bovw(data)
    # data = extract_features(data)
    # for i in range(len(data)):
    #     cv2.imshow(data[i]['label'],data[i]['image'])
    #     cv2.waitKey(0)
    # img = cv2.imread('train/images/road0.png')
    # cv2.rectangle(img,(98,62),(208,232),(0,255,0))
    # cv2.imshow('test',img)
    # cv2.waitKey(0)
    return


if __name__ == '__main__':
    main()