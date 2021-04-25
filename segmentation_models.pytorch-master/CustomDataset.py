import torch
import os
import pickle
import numpy as np
import cv2
from torch.utils.data import Dataset

class DepthDataset(Dataset):
    def __init__(self, folder_list, transform=None):
        self.transform = transform
        self.folder_list = folder_list
        self.fileNames = []
        for every in folder_list:
            self.fileNames.append(os.listdir(every))

    def __len__(self):
        sum = 0
        for every in self.fileNames:
            sum += len(every)//3
        return sum
    
    def __getitem__(self, index):    
        tmp = 0
        i = 0
        folder_name = None
        for every in self.fileNames:
            if (index<tmp + len(every)//3):
                folder_name = self.folder_list[i]
                break
            else:
                tmp += len(every)//3
                i+=1
            
        file = open(folder_name+"/"+str(index-tmp)+".configs", "rb")
        configs = pickle.load(file)
        file.close()
        file = open(folder_name+"/"+str(index-tmp)+".color", "rb")
        bg_removed = pickle.load(file)
        file.close()
        file = open(folder_name+"/"+str(index-tmp)+".depth", "rb")
        depth_image = pickle.load(file)
        file.close()
        
        depth_scale = configs[5]
        clipping_distance_in_meters = 1 #1 meter
        clipping_distance = clipping_distance_in_meters / depth_scale

        bg_removed = bg_removed.reshape((int(configs[0]),int(configs[1]),int(configs[2])))
        depth_image = depth_image.reshape((int(configs[3]),int(configs[4]),1))
        label_image = np.zeros(int(configs[3])*int(configs[4]), dtype=np.int8)
        label_image = label_image.reshape((int(configs[3]),int(configs[4]),1))
        
        
        
        for j in range(6, len(configs)):
            pts = np.array([[int(configs[j][0]),int(configs[j][1])],[int(configs[j][2]),int(configs[j][3])],[int(configs[j][4]),int(configs[j][5])],[int(configs[j][6]),int(configs[j][7])]], np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.fillPoly(label_image, [pts], color=(1+int(configs[j][8])))
            cv2.polylines(label_image,[pts], True, (1+int(configs[j][8])),15)
        
        picture = np.zeros(480*640*4)
        picture = picture.reshape((480,640,4))
        label = np.zeros(480*640*1)
        label = label.reshape((1,480,640))
        for k in range(480):
            for j in range(640):
                picture[k][j][0] = bg_removed[k][j][0]
                picture[k][j][1] = bg_removed[k][j][1]
                picture[k][j][2] = bg_removed[k][j][2]
                picture[k][j][3] = depth_image[k][j]
                label[0][k][j] = label_image[k][j]

        return self.transform(picture) if self.transform else picture, label, index
        
