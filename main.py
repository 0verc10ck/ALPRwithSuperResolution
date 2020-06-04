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
cnt = 0 


def init():
    global list_img, file
    list_img = np.array(os.listdir('./data/dataset/'))
    os.makedirs("./data/preprocessed_img", exist_ok=True)
    os.makedirs("./data/sr_img", exist_ok=True)
    file = open("result.txt", 'w', encoding="utf-8")

    
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
        path = './data/dataset/' + list_img[i]
        #detections = (darknet.performDetect(path, 0.25, configPath, weightPath, metaPath, False, False))
        detections = darknet.detect(netMain, metaMain, path.encode("ascii"), 0.25)
        SuperResolution(list_img[i], preprocess(path, detections))
        
#center_x, center_y, w, h
def preprocess(path, detections):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    f_name = path.split('/')[-1]
    #create coord
    xmin = int(detections[0][2][0]-detections[0][2][2]/2)
    xmax = int(detections[0][2][0]+detections[0][2][2]/2)
    ymin = int(detections[0][2][1]-detections[0][2][3]/2)
    ymax = int(detections[0][2][1]+detections[0][2][3]/2)
    
    cut = img[ymin:ymax, xmin:xmax].copy()
    gray=cv2.cvtColor(cut,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,90,255,0)
    contours,_ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    if(len(areas)!=0):
        max_index = np.argmax(areas)
        cnt=contours[max_index]
        x,y,w,h = cv2.boundingRect(cnt)
        coord = np.float32([
            [x, y],
            [x+w, y],
            [x, y+h],
            [x+w, y+h]
        ])
            
        dst = np.float32([
            [0,0],
            [350, 0],
            [0, 100],
            [350, 100],
        ])
    
        matrix = cv2.getPerspectiveTransform(coord, dst)
        img_t = cv2.warpPerspective(cut, matrix, (350, 100))
        
    else:
        img_t = cut
    
    del img, cut, gray
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
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    
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
    result  = OCR(path)
    mse, psnr = PSNR(oripath, path)
    save_result(f_name, result, mse, psnr)
    
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

    MIN_AREA = 250
    MIN_WIDTH, MIN_HEIGHT = 50, 150
    MAX_WIDTH = 300
    
    possible_contours = []

    cnt = 0
    for d in contours_dict:
        area = d['w'] * d['h']
    
        if area > MIN_AREA and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT and d['w'] < MAX_WIDTH:
            d['idx'] = cnt
            cnt += 1
            possible_contours.append(d)
    
    
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)
    xmax = 0
    xmin = 1400
    ymax = 0
    ymin = 400
    if len(possible_contours) > 3:
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
            xmax += 20
        if ymax < 380:
            ymax += 20
        
        img = ori[ymin:ymax, xmin:xmax].copy()
    else :
        img = ori
        
    string = ''
    try:
        string = pytesseract.image_to_string(img, lang='kor', config='--oem 1 --psm 7')
        string = postprocess(string)
        
    except IndexError:
            pass
    return string

def postprocess(string):
    string = string.replace(" ", "")
    string = string.replace(":", "")
    string = string.replace(".", "")
    string = string.replace(";", "")
    string = string.replace("*", "")
    string = string.replace("\"", "")
    
    string = string.replace("가", "ga")
    string = string.replace("나", "na")
    string = string.replace("다", "da")
    string = string.replace("라", "ra")
    string = string.replace("마", "ma")
    string = string.replace("바", "ba")
    string = string.replace("사", "sa")
    string = string.replace("자", "ja")
    string = string.replace("아", "a")
    
    string = string.replace("거", "geo")
    string = string.replace("너", "neo")
    string = string.replace("더", "deo")
    string = string.replace("러", "reo")
    string = string.replace("머", "meo")
    string = string.replace("버", "beo")
    string = string.replace("서", "seo")
    string = string.replace("저", "jeo")
    string = string.replace("어", "eo")
    
    string = string.replace("고", "go")
    string = string.replace("노", "no")
    string = string.replace("도", "do")
    string = string.replace("로", "ro")
    string = string.replace("모", "mo")
    string = string.replace("보", "bo")
    string = string.replace("소", "so")
    string = string.replace("조", "jo")
    string = string.replace("오", "o")
    
    string = string.replace("구", "gu")
    string = string.replace("누", "nu")
    string = string.replace("두", "du")
    string = string.replace("루", "ru")
    string = string.replace("무", "mu")
    string = string.replace("부", "bu")
    string = string.replace("수", "su")
    string = string.replace("주", "ju")
    string = string.replace("우", "u")
    
    string = string.replace("배", "bae")
    string = string.replace("하", "ha")
    string = string.replace("허", "heo")
    string = string.replace("호", "ho")
    
    return string

def save_result(f_name, result,mse, psnr):
    global cnt
    print(cnt,"th step")
    #print(f_name, result, mse, psnr)
    save = "%s, %s, %s, %s" % (f_name,result,mse,psnr)
    file.write(save) 
    cnt += 1


from matplotlib import pyplot as plt
if __name__ == "__main__":
    init()
    Detector()
