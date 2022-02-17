import os
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import xml.etree.ElementTree as ET
from pathlib import Path

def load_from_xml(path):
    data = []
    for i in os.listdir(path+'annotations/'):
        tree = ET.parse(path+'annotations/'+i)
        filename = tree.find('filename')
        print(filename.text)
        objects = tree.findall('object')
        print(len(objects))
        size = tree.find('size')
        height = size.find('height')
        width = size.find('width')
        for i in objects:
            x_min = i.find('bndbox/xmin')
            y_min = i.find('bndbox/ymin')
            x_max = i.find('bndbox/xmax')
            y_max = i.find('bndbox/ymax')
            print(x_min.text,y_min.text,x_max.text,y_max.text)
            image = cv2.imread(path+'images/'+filename.text)
            image_cropped = image[int(y_min.text):int(y_max.text), int(x_min.text):int(x_max.text)]
            if str(i.find('name').text) == 'speedlimit' and (int(y_max.text)-int(y_min.text))*10 > int(height.text) and (int(x_max.text)-int(x_min.text))*10 > int(width.text):
                data.append({'image': image_cropped, 'label': str(i.find('name').text)})
            else:
                data.append({'image': image_cropped, 'label': 'other'})
    return data

def load_from_input(path):
    data = []
    quantity = input()
    for i in range(int(quantity)):
        filename = input()
        img_to_classify = input()
        for i in range(int(img_to_classify)):
            temp = input()
            cord = temp.split()
            image = cv2.imread(path + 'images/' + filename)
            image_cropped = image[int(cord[2]):int(cord[3]), int(cord[0]):int(cord[1])]
            data.append({'image': image_cropped, 'label': ''})
    return data

def learn_bovw(data):
    size = 128
    bow = cv2.BOWKMeansTrainer(size)

    sift = cv2.SIFT_create()
    for i in data:
        kpts = sift.detect(i['image'], None)
        kpts, desc = sift.compute(i['image'], kpts)
        if desc is not None:
            bow.add(desc)
    vocabulary = bow.cluster()
    return vocabulary

def extract_features(data,vocabulary):
    sift = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher_create()
    bow = cv2.BOWImgDescriptorExtractor(sift, flann)
    bow.setVocabulary(vocabulary)

    for sample in data:
        kpts = sift.detect(sample['image'], None)
        imgDes = bow.compute(sample['image'], kpts)
        if imgDes is not None:
            sample.update({'desc': imgDes})
        else:
            sample.update({'desc': np.zeros((1, 128))})
    return data

def train(data):
    clf = RandomForestClassifier(128)
    x_matrix = np.empty((1, 128))
    y_vector = []
    for sample in data:
        y_vector.append(sample['label'])
        x_matrix = np.vstack((x_matrix, sample['desc']))
    clf.fit(x_matrix[1:], y_vector)
    return clf

def predict(rf, data):
    for sample in data:
        sample.update({'label_pred': rf.predict(sample['desc'])[0]})
    return data

def evaluate(data):
    y_pred = []
    y_real = []
    for sample in data:
        y_pred.append(sample['label_pred'])
        y_real.append(sample['label'])

    confusion = confusion_matrix(y_real, y_pred)
    print(confusion)
    _TPa, _Eba, _Eab, _TPb = confusion.ravel()
    accuracy = 100 * (_TPa + _TPb ) / (_TPa + _Eba + _Eab + _TPb)
    print("accuracy =", round(accuracy, 2), "%")
    return

def main():
    main = Path('main.py')
    main_path = main.parent.absolute()
    absolute_path=main_path.parent
    temp = input()
    if temp == 'detect':
        train_data = load_from_xml(str(absolute_path) + '/train/')
        test_data = load_from_xml(str(absolute_path) + '/test/')
        vocabulary = learn_bovw(train_data)
        train_data = extract_features(train_data,vocabulary)
        rf = train(train_data)
        test_data = extract_features(test_data,vocabulary)
        test_data = predict(rf, test_data)
        evaluate(test_data)
    elif temp == 'classify':
        input_data = load_from_input(str(absolute_path) + '/test/')
        train_data = load_from_xml(str(absolute_path) + '/train/')
        vocabulary = learn_bovw(train_data)
        train_data = extract_features(train_data, vocabulary)
        rf = train(train_data)
        input_data = extract_features(input_data, vocabulary)
        input_data = predict(rf, input_data)
        for i in range(len(input_data)):
            print(input_data[i]['label_pred'])
    return


if __name__ == '__main__':
    main()