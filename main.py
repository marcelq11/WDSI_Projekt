

import os
import random
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pandas
import xml.etree.ElementTree as ET

def main():
    data = []
    for i in range(800):
        z=0
        temp = []
        tree = ET.parse('train/annotations/road'+str(i)+'.xml')
        root = tree.getroot()
        for child in root:
            if child.tag == 'filename':
                print(child.text)
            if child.tag == 'object':
                if root[4+z][0].text == 'speedlimit':
                    text = str(root[4+z][5][0].text)+' '+str(root[4+z][5][1].text)+' '+str(root[4+z][5][2].text)+' '+str(root[4+z][5][3].text)
                    temp.append(text)
                z+=1
        print(len(temp))
        for i in range(len(temp)):
            print(temp[i])
        print('\n')
    return


if __name__ == '__main__':
    main()