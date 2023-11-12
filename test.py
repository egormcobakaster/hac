import torch
from models.experimental import attempt_load
import numpy as np
from PIL import Image
from utils.general import non_max_suppression
from models_anti import *
from tsn_predict import *
import os
import matplotlib.pyplot as plt
import cv2

from client import get_thresholdtable_from_fpr, get_tpr_from_threshold

PATH_DATA = '/Users/egorperelygin/Downloads/DataSet/NUAA/spoof/0001'

face_model = attempt_load("./yolov7s-face.pt", map_location="cpu")
list_im = []
for address, dirs, files in os.walk(PATH_DATA):
    for name in files:
        if name[-1] == 'g':
            list_im.append(os.path.join(address, name))

pt_model = TSNPredictor()
scores = []
labels = []
count = 0
list_im = np.array(list_im)
np.random.shuffle(list_im)
count_spoof = 0
count_live = 0
for im_path in list_im:
    
    if count % 200 == 0:
        print(count)
    count += 1
    orgimg = cv2.imread(im_path)
    orgimg = cv2.cvtColor(orgimg, cv2.COLOR_BGR2RGB)
    # plt.imshow(orgimg)
    # name_im = str(count)
    # plt.savefig('/Users/egorperelygin/ex/' + name_im + 'foo.png')
    
    orgimg_t = torch.tensor(np.transpose(orgimg,(2,0,1))).unsqueeze(0) / 255.
    outs = face_model(orgimg_t)[0]
    box = non_max_suppression(outs)[0][0,:4].int()

    real_h,real_w,c = orgimg.shape
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]
    h = y2 - y1
    w = x2 - x1
    x1 = box[0] - w // 6
    y1 = box[1] - h // 6
    x2 = box[2] + w // 6
    y2 = box[3] + h // 6
    y1 = 0 if y1 < 0 else y1
    x1 = 0 if x1 < 0 else x1 
    y2 = real_h if y2 > real_h else y2
    x2 = real_w if x2 > real_w else x2
    crop_im = orgimg[y1:y2,x1:x2,:]

    #plt.imshow(crop_im)
    #name_im = str(count)
    #plt.savefig('/Users/egorperelygin/' + name_im + 'foo.png')


    #labels.append(int(im_path.split('/')[-2] == 'spoof'))
    out = pt_model.predict([crop_im])
    scores.append(out[0,0])

with open('./out1.txt', 'w') as file:
    for i in range(len(scores)):
        file.write(list_im[i] + " " + str(scores[i]) + '\n')

# fpr_list = [0.01, 0.005, 0.001]
# threshold_list = get_thresholdtable_from_fpr(scores,labels, fpr_list)
# tpr_list = get_tpr_from_threshold(scores,labels, threshold_list)
    
# # Show the result into score_path/score.txt  
# print('TPR@FPR=10E-3: {}\n'.format(tpr_list[0]))
# print('TPR@FPR=5E-3: {}\n'.format(tpr_list[1]))
# print('TPR@FPR=10E-4: {}\n'.format(tpr_list[2]))
