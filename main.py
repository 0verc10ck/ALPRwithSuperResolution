from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import darknet
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19


netMain = None
metaMain = None
altNames = None


def init():
    global list_img
    list_img = np.array(os.listdir('./data/test'))
    os.makedirs("./data/preprocessed_img", exist_ok=True)
    os.makedirs("./data/sr_img", exist_ok=True)

#Yolo Detector
def Detector():
    global metaMain, netMain, altNames
    configPath = "./yolov3.cfg"
    weightPath = "./yolov3.weights"
    metaPath = "./data/obj.data"

    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))    
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass

    darknet_image = darknet.make_image(darknet.network_width(netMain), darknet.network_height(netMain),3)
    for i in range(0, len(list_img)):  
        path = './data/testn/' + list_img[i]
        detections = (darknet.performDetect(path, 0.25, configPath, weightPath, metaPath, False, False))
        preprocess(path, detections)

     
        
def preprocess(path, detections):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    f_name = path.split('/')[-1]
    #create coord
    x_min = int(detections[0][2][0]-detections[0][2][2]/2)
    x_max = int(detections[0][2][0]+detections[0][2][2]/2)
    y_min = int(detections[0][2][1]-detections[0][2][3]/2)
    y_max = int(detections[0][2][1]+detections[0][2][3]/2)
    
    coord = np.float32([
        [x_min, y_min],
        [x_max, y_min],
        [x_min, y_max],
        [x_max, y_max]
    ])
    
    #transformation

    dst2 = np.float32([
        [0,0],
        [350, 0],
        [0, 100],
        [350, 100],
    ])
    
    matrix = cv2.getPerspectiveTransform(coord, dst2)
    img_t = cv2.warpPerspective(img, matrix, (350, 100))
    
    #save
    transpath = "./data/preprocessed_img"
    try:
        cv2.imwrite(os.path.join(transpath,f_name),img_t)
    
    except ValueError:
        print("Cannot save file ", f_name, "\n")
        pass
    
if __name__ == "__main__":
    init()
    Detector()

