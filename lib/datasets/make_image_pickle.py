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
        self.image_len = 10000
        self.save_root = f'/content/drive/MyDrive/VSE/image_list_npy_{args.number}.npy'
    
    def __getitem__(self, index):
        
        path = f"thumnail_image_{index + args.number}.png"
        im_in = np.array( imread(os.path.join(self.image_root, path),pilmode='RGB') )
        #im_in = np.zeros((1,12,1))
        return im_in, index
    
    def __len__(self):
        return self.image_len

def collate_fun(data):
    image, index = zip(*data)
    image_return = []
    index_return = []
    for img in image:
        image_return.append(img)
    for idx in index:
        index_return.append(idx)
    
    return image_return , index_return

def start():
    customdataset = CustomImageDataset()
    CustomLoader = torch.utils.data.DataLoader(dataset=customdataset,
                                                  batch_size=128,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=8,
                                                  drop_last=False,
                                                  collate_fn=collate_fun)
    image_list = [None] * customdataset.image_len
    
    for i, (im_numpy_array, im_index) in tqdm(enumerate(CustomLoader)):
        for cur_img, cur_index in zip(im_numpy_array,im_index):
            image_list[cur_index] = cur_img
        del im_numpy_array, im_index
    print("loading done")
    image_list = np.array(image_list, dtype=object)
    print("saving...")
    np.save(customdataset.save_root, image_list)


import argparse

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--number', type=int, help='start index')
    args = parser.parse_args()
    start()