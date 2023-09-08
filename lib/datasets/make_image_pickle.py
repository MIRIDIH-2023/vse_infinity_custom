import torch
import torch.utils.data as data
import os
import os.path as osp
import numpy as np
from imageio import imread
import random
import json
import cv2
from PIL import Image
import pickle
from tqdm import tqdm

class CustomImageDataset(data.Dataset):
    def __init__(self):
        super().__init__()
        self.image_root = '/content/drive/MyDrive/images'
        self.image_len = 45000
        self.save_root = '/content/drive/MyDrive/VSE/image_list_npy.npy'
    
    def __getitem__(self, index):
        path = f"thumnail_image_{index}.png"
        im_in = np.array( imread(os.path.join(self.image_root, path),pilmode='RGB') )
        return im_in, index
    
    def __len__(self):
        return self.image_len

def start():
    customdataset = CustomImageDataset()
    CustomLoader = torch.utils.data.DataLoader(dataset=customdataset,
                                                  batch_size=128,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=12,
                                                  drop_last=False)
    image_list = [None] * customdataset.image_len
    for i, (im_numpy_array, im_index) in tqdm(enumerate(CustomLoader)):
        for cur_img, cur_index in zip(im_numpy_array,im_index):
            image_list[cur_index] = cur_img
    
    np.save(customdataset.save_root, image_list)

if __name__=="__main__":
    start()