from ctypes import *
from PIL import Image
from torchvision.models import vgg19
from models import GeneratorRRDB
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
from datasets import denormalize, mean, std

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


def PSNR(img1, img2):
    mse = np.mean(( img1 - img2) ** 2)
    print("mse : ", mse)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20* math.log10(PIXEL_MAX / math.sqrt(mse))

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
    save_image(sr_image, path)
    #OCR(path)
    

if __name__ == "__main__":
    init()
    Detector()