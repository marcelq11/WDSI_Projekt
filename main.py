import os
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import xml.etree.ElementTree as ET
from pathlib import Path

# TODO Jakość kodu i raport (4/4)
# TODO Raport troche skapy.

# TODO Skuteczność klasyfikacji 0.712 (2.5/4)
# TODO [0.00, 0.50) - 0.0
# TODO [0.50, 0.55) - 0.5
# TODO [0.55, 0.60) - 1.0
# TODO [0.60, 0.65) - 1.5
# TODO [0.65, 0.70) - 2.0
# TODO [0.70, 0.75) - 2.5
# TODO [0.75, 0.80) - 3.0
# TODO [0.80, 0.85) - 3.5
# TODO [0.85, 1.00) - 4.0


# TODO Skuteczność detekcji (/2)


def load_from_xml(path):
    data = []
    for i in os.listdir(path+'annotations/'):
        tree = ET.parse(path+'annotations/'+i)
        filename = tree.find('filename')
        objects = tree.findall('object')
        size = tree.find('size')
        height = size.find('height')
        width = size.find('width')
        for i in objects:
            x_min = i.find('bndbox/xmin')
            y_min = i.find('bndbox/ymin')
            x_max = i.find('bndbox/xmax')
            y_max = i.find('bndbox/ymax')
            image = cv2.imread(path+'images/'+filename.text)
            image_cropped = image[int(y_min.text):int(y_max.text), int(x_min.text):int(x_max.text)]
            if str(i.find('name').text) == 'speedlimit' and (int(y_max.text)-int(y_min.text))*10 > int(height.text) and (int(x_max.text)-int(x_min.text))*10 > int(width.text):
                data.append({'image': image_cropped, 'label': str(i.find('name').text)})
            else:
                data.append({'image': image_cropped, 'label': 'other'})
    return data

def load_png(path):
    data = []
    for i in os.listdir(path + 'images/'):
        image = cv2.imread(path + 'images/' + i)
        data.append({'image': image, 'label': ''})
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
            # TODO Format to "xmin xmax ymin ymax".
            image_cropped = image[int(cord[1]):int(cord[3]), int(cord[0]):int(cord[2])]
            data.append({'image': image_cropped, 'label': ''})
    return data

def learn(data):
    size = 128
    bow = cv2.BOWKMeansTrainer(size)
    sift = cv2.SIFT_create()
    for i in data:
        key_points, des = sift.detectAndCompute(i['image'], None)
        if des is not None:
            bow.add(des)
    vocabulary = bow.cluster()
    return vocabulary

def extract(data,vocabulary):
    sift = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher_create()
    bow = cv2.BOWImgDescriptorExtractor(sift, flann)
    bow.setVocabulary(vocabulary)

    for i in data:
        key_points = sift.detect(i['image'], None)
        img_des = bow.compute(i['image'], key_points)
        if img_des is not None:
            i.update({'desc': img_des})
        else:
            # TODO Lepiej w ogole pominac takie przypadki.
            i.update({'desc': np.zeros((1, 128))})
    return data

def train(data):
    clf = RandomForestClassifier(128)
    # TODO Mozna tez zrobic "np.empty((0, 128))".
    x = np.empty((1, 128))
    y = []
    for i in data:
        y.append(i['label'])
        x = np.vstack((x, i['desc']))
    clf.fit(x[1:], y)
    return clf

def predict(rf, data):
    for i in data:
        i.update({'label_pred': rf.predict(i['desc'])[0]})
    return data

def main():

    main = Path('main.py')
    main_path = main.parent.absolute()
    absolute_path=main_path.parent
    temp = input()
    if temp == 'detect':
        train_data = load_from_xml(str(absolute_path) + '/train/')
        # test_data_check = load_from_xml(str(absolute_path) + '/test/')
        test_data = load_png(str(absolute_path) + '/test/')
        vocabulary = learn(train_data)
        train_data = extract(train_data,vocabulary)
        rf = train(train_data)
        test_data = extract(test_data, vocabulary)


    elif temp == 'classify':
        input_data = load_from_input(str(absolute_path) + '/test/')
        train_data = load_from_xml(str(absolute_path) + '/train/')
        vocabulary = learn(train_data)
        train_data = extract(train_data, vocabulary)
        rf = train(train_data)
        input_data = extract(input_data, vocabulary)
        input_data = predict(rf, input_data)
        for i in range(len(input_data)):
            print(input_data[i]['label_pred'])
    return


if __name__ == '__main__':
    main()