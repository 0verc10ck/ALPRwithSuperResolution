import os
import cv2
import json
import numpy as np
import datetime
from matplotlib import pyplot as plt

#initializing test.txt
def init_test_txt():
    
    #dir of darknet/data/ . implement soon..
    #adr = './data'
    #preprocessing.py file will be place at /darknet
    #initiate test.txt
    list_img = np.array(os.listdir('./data/img'))

    f=open('./data/test.txt','w')
    os.makedirs("./data/processed_img", exist_ok=True)
    for fname in list_img:
        f.write("./data/img/"+fname+"\n")
    f.close()

# Preprocessing
def preprocessing():

    with open("./result.json", 'rb') as f:
        json_data = np.array(json.load(f))

    for i in range(0,len(json_data)):
        try:
            file_dir = json_data[i]['filename']
            cx_pos = json_data[i]['objects'][0]['relative_coordinates']['center_x']
            cy_pos = json_data[i]['objects'][0]['relative_coordinates']['center_y']
            w = json_data[i]['objects'][0]['relative_coordinates']['width']
            h = json_data[i]['objects'][0]['relative_coordinates']['height']

            info_pos = np.array([[cx_pos,w],[cy_pos,h]])
            print("processing ", json_data[i]['filename'], "\n")
            crop_and_save(info_pos,file_dir,i)

        except IndexError:
            pass

#save
def crop_and_save(pos, file_dir,idx):

    img = cv2.imread(file_dir, cv2.IMREAD_GRAYSCALE)
    size = np.array(np.array(img).shape)
    
    #calulate coordinates
    pos[0]*=size[1]
    pos[1]*=size[0]

    x_min = int(pos[0][0]-pos[0][1]/2)
    x_max = int(pos[0][0]+pos[0][1]/2)
    y_min = int(pos[1][0]-pos[1][1]/2)
    y_max = int(pos[1][0]+pos[1][1]/2)
    
    #coordinate create
    coord = np.float32([
        [x_min, y_min],
        [x_max, y_min],
        [x_min, y_max],
        [x_max, y_max]
    ])
    
    #get file name
    f_name = file_dir.split('/')[-1]
    
    #tranformation
    img = Transformation(coord, img)
    
    #save

    path = "./data/processed_img/"
    try:
        cv2.imwrite(os.path.join(path,f_name),img)
    
    except ValueError:
        print("Cannot save file ", f_name, "\n")
        pass

###############################main###########################


def Transformation(coord, img):
    #ratio of licnese plate is 2:1 on the old and 5:1 on the new
    #we used 3.5:1
    
    dst = np.float32([
        [0,0],
        [img.shape[1], 0],
        [0, img.shape[0]],
        [img.shape[1], img.shape[0]],
    ])
    
    dst2 = np.float32([
        [0,0],
        [350, 0],
        [0, 100],
        [350, 100],
    ])

    
    matrix = cv2.getPerspectiveTransform(coord, dst2)
    img_t2 = cv2.warpPerspective(img, matrix, (350, 100))
    
    return img_t2
    

if __name__ == "__main__":
    init_test_txt()
    #Detector()
    preprocessing()

