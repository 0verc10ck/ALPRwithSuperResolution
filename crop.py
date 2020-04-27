import os
import cv2
import json
import numpy as np
import datetime

#initializing test.txt
def init_test_txt(adr):
    
    #dir of darknet/data/ . implement soon..
    #adr = './data'

    #initiate test.txt
    list_img = np.array(os.listdir('/home/cdp8/Desktop/darknet/data/img'))

    f=open('/home/cdp8/Desktop/darknet/data/test.txt','w')

    for l in list_img:
        f.write("/home/cdp8/Desktop/darknet/data/img/"+l+"\n")
    f.close()

# Preprocessing
def preprocessing():

    with open("/home/cdp8/Desktop/darknet/result.json", 'rb') as f:
        json_data = np.array(json.load(f))

    for i in range(0,len(json_data)):
        try:
            file_dir = json_data[i]['filename']
            cx_pos = json_data[i]['objects'][0]['relative_coordinates']['center_x']
            cy_pos = json_data[i]['objects'][0]['relative_coordinates']['center_y']
            w = json_data[i]['objects'][0]['relative_coordinates']['width']
            h = json_data[i]['objects'][0]['relative_coordinates']['height']

            info_pos = np.array([[cx_pos,w],[cy_pos,h]])
            preprocess_img(info_pos,file_dir,i)

        except IndexError:
            pass

#crop and save image
def preprocess_img(pos, file_dir,idx):

    img = cv2.imread(file_dir)
    size = np.array(np.array(img).shape)
    
    #calulate coordinates
    pos[0]*=size[1]
    pos[1]*=size[0]

    x_min = int(pos[0][0]-pos[0][1]/2)
    x_max = int(pos[0][0]+pos[0][1]/2)
    y_min = int(pos[1][0]-pos[1][1]/2)
    y_max = int(pos[1][0]+pos[1][1]/2)
    
    #crop
    c_img = img[y_min:y_max, x_min:x_max]

    #naming preventing to overlapping
    b_name = "_img.jpg"
    path = "/home/cdp8/Desktop/cdp8/data"
    prefix = datetime.datetime.now().strftime("%y%m%d%H%M%S_")
    f_name = str(idx).join([prefix,b_name])
    
    #save
    cv2.imwrite(os.path.join(path,f_name),c_img)

###############################main###########################

if __name__ == "__main__":
    init_test_txt('./')
    #Detector()
    preprocessing()
