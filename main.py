from ctypes import *
from PIL import Image
from torchvision.models import vgg19
from models import GeneratorRRDB
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
from datasets import denormalize, mean, std
from skimage.segmentation import clear_border
from imutils import contours

import math
import random
import os
import cv2
import numpy as np
import darknet
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pytesseract
import imutils

netMain = None
metaMain = None
altNames = None 


def init():
    global list_img
    list_img = np.array(os.listdir('./data/testn'))
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
        SuperResolution(list_img[i], preprocess(path, detections))

     
        
#center_x, center_y, w, h
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
    del img
    #save
    transpath = "./data/preprocessed_img"
    try:
        cv2.imwrite(os.path.join(transpath,f_name), img_t)
    
    except ValueError:
        print("Cannot save file ", f_name, "\n")
        pass    
    return img_t


def PSNR(oripath, path):
    img1 =  cv2.imread(oripath)
    img2 = cv2.imread(path)
    img1 = cv2.resize(img1, dsize=(1400, 400),interpolation=cv2.INTER_AREA)
    mse = np.mean((img1 - img2) ** 2)
    #print("mse : ", mse)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    psnr = 20* math.log10(PIXEL_MAX / math.sqrt(mse))
    return mse, psnr


def SuperResolution(f_name, ori):
    pth = "./generator.pth"
    channels = 3
    residual_blocks = 23
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    # Define model and load model checkpoint
    generator = GeneratorRRDB(channels, filters=64, num_res_blocks=residual_blocks).to(device)
    generator.load_state_dict(torch.load(pth))
    generator.eval()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    # Prepare input
    
    image_tensor = Variable(transform(ori)).to(device).unsqueeze(0)

    # Upsample image

    with torch.no_grad():
        sr_image = denormalize(generator(image_tensor)).cpu()
    
    # Save image
    path = os.path.join("./data/sr_img/",f_name)
    oripath = os.path.join("./data/preprocessed_img/",f_name)
    save_image(sr_image, path)
    mse, psnr = PSNR(oripath, path)
    string  = OCR(path)

    
def OCR(path):
    img = cv2.imread(path)
    height, width, channel = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    imgTopHat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, structuringElement)
    imgGrayscalePlusTopHat = cv2.add(img, imgTopHat)
    img = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)
    
    img = cv2.GaussianBlur(img, ksize=(9, 9), sigmaX=0)
    ori =img.copy()

    img = cv2.adaptiveThreshold(
        img, 
        maxValue=255, 
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        thresholdType=cv2.THRESH_BINARY_INV, 
        blockSize=255, 
        C=10
    )
    
    contours, _ = cv2.findContours(
        img, 
        mode=cv2.RETR_LIST, 
        method=cv2.CHAIN_APPROX_SIMPLE
    )

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)
    contours_dict = []
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(temp_result, pt1=(x,y), pt2=(x+w, y+h), color=(255,255,255))
    
        contours_dict.append({
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x+(w/2),
            'cy': y+(h/2)
        })
    
    MIN_AREA = 1000
    MIN_WIDTH, MIN_HEIGHT = 50, 150
    MAX_WIDTH = 500

    possible_contours = []

    cnt = 0
    for d in contours_dict:
        area = d['w'] * d['h']
    
        if area > MIN_AREA and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT and d['w'] < MAX_WIDTH :
            d['idx'] = cnt
            cnt += 1
            possible_contours.append(d)
        
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)
    
    xmax = 0
    xmin = 1400
    ymax = 0
    ymin = 400
    for d in possible_contours:
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)
        if xmin > d['x'] :
            xmin = d['x']
        if xmax < d['x']+d['w'] :
            xmax = d['x']+d['w']
        if ymin > d['y'] :
            ymin = d['y']
        if ymax < d['y']+d['h'] :
            ymax = d['y']+d['h']
            
    if xmin > 20:
        xmin -= 20
    if ymin > 20:
        ymin -= 20
    if xmax < 1380:
        xmax +=20
    if ymax < 380:
        ymax += 20
    
    img = ori[ymin:ymax, xmin:xmax].copy()
    plt.imshow(img, cmap='gray')
    plt.show()  
    string = pytesseract.image_to_string(img, lang='kor', config='--oem 1 --psm 7')
    return string


if __name__ == "__main__":
    init()
    Detector()

