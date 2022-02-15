

import os
import random
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pandas
import xml.etree.ElementTree as ET
from pathlib import Path

def load_data(path,train_quantity):
    data = []
    for i in range(train_quantity):
        tree = ET.parse(path+'annotations\\road'+str(i)+'.xml')
        filename = tree.find('filename')
        print(filename.text)
        objects = tree.findall('object')
        print(len(objects))
        for i in objects:
            x_min = i.find('bndbox/xmin')
            y_min = i.find('bndbox/ymin')
            x_max = i.find('bndbox/xmax')
            y_max = i.find('bndbox/ymax')
            print(x_min.text,y_min.text,x_max.text,y_max.text)
            image = cv2.imread(path+'images\\'+filename.text)
            image_cropped = image[int(y_min.text):int(y_max.text), int(x_min.text):int(x_max.text)]
            data.append({'image': image_cropped, 'label': i.find('name').text})
    return data

def main():
    train_quantity = 5
    main = Path('main.py')
    main_path = main.parent.absolute()
    absolute_path=main_path.parent
    data = load_data(str(absolute_path)+'\\train\\',train_quantity)
    for i in range(len(data)):
        cv2.imshow(data[i]['label'],data[i]['image'])
        cv2.waitKey(0)
    # img = cv2.imread('train/images/road0.png')
    # cv2.rectangle(img,(98,62),(208,232),(0,255,0))
    # cv2.imshow('test',img)
    # cv2.waitKey(0)
    return


if __name__ == '__main__':
    main()