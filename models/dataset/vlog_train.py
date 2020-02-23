from __future__ import print_function, absolute_import

import os
import numpy as np
import json
import random
import math

import torch
import torch.utils.data as data
import random

from utils.imutils2 import *
from utils.transforms import *
import torchvision.transforms as transforms

import scipy.io as sio
import scipy.misc
from geotnf.transformation import GeometricTnf

import torchvision.transforms as transforms

class VlogSet(data.Dataset):
    def __init__(self, params, is_train=True, frame_gap=1, augment=['crop', 'flip', 'frame_gap']):

        self.filelist = params['filelist']
        self.batchSize = params['batchSize']
        self.imgSize = params['imgSize']
        self.cropSize = params['cropSize']
        self.cropSize2 = params['cropSize2']
        self.videoLen = params['videoLen']
        # prediction distance, how many frames far away
        self.predDistance = params['predDistance']
        # offset x,y parameters
        self.offset = params['offset']
        # gridSize = 3
        self.gridSize = params['gridSize']

        self.is_train = is_train
        self.frame_gap = frame_gap

        self.augment = augment

        f = open(self.filelist, 'r')
        self.jpgfiles = []
        self.fnums = []

        for line in f:
            rows = line.split()
            jpgfile = rows[0]   # sth like /YOUR_DATAFOLDER/B/X/g/v_afgUDkDaBXg/010/
            fnum = int(rows[1]) # how many images in that folder
            # fnum = len(os.listdir(fname))

            self.jpgfiles.append(jpgfile)
            self.fnums.append(fnum)

        f.close()
        # This geometric transformation --> output image patch tensor with shape [2, 80, 80]
        self.geometricTnf = GeometricTnf('affine', out_h=params['cropSize2'], out_w=params['cropSize2'], use_cuda = False)

    def cropimg(self, img, offset_x, offset_y, cropsize):

        img = im_to_numpy(img)
        cropim = np.zeros([cropsize, cropsize, 3])
        cropim[:, :, :] = img[offset_y: offset_y + cropsize, offset_x: offset_x + cropsize, :]
        cropim = im_to_torch(cropim)

        return cropim

    def cropimg_np(self, img, offset_x, offset_y, cropsize):

        cropim = np.zeros([cropsize, cropsize])
        cropim[:, :] = img[offset_y: offset_y + cropsize, offset_x: offset_x + cropsize]

        return cropim

    def processflow(self, flow):
        boundnum = 60
        flow = flow.astype(np.float)
        flow = flow / 255.0
        flow = flow * boundnum * 2
        flow = flow - boundnum

        flow = np.abs(flow)

        return flow


    def __getitem__(self, index):

        folder_path = self.jpgfiles[index] # self.jpgfiles: .jpg folder list
        fnum = self.fnums[index]           # # of .jpg files in that list
        # By default videoLen==4; cropSize==240; 
        imgs = torch.Tensor(self.videoLen, 3, self.cropSize, self.cropSize) #[4,3,240,240]
        imgs_target  = torch.Tensor(2, 3, self.cropSize, self.cropSize)     #[2,3,240,240]
        patch_target = torch.Tensor(2, 3, self.cropSize, self.cropSize)     #[2,3,240,240]
        future_imgs  = torch.Tensor(2, 3, self.cropSize, self.cropSize)     #[2,3,240,240] so why it is 2? in the Batchsize position!
        # patch_target2 = torch.Tensor(2, 3, self.cropSize2, self.cropSize2)

        toflip = False
        if random.random() <= 0.5:
            toflip = True

        frame_gap = self.frame_gap # 2
        current_len = (self.videoLen  + self.predDistance) * frame_gap # videoLen=4; predDistance=4; framegap=2; --> current_len=16
        startframe = 0
        future_idx = current_len # 16

        if fnum >= current_len: # if number of .jpg files larger then 16; suppose fnum=25
            diffnum = fnum - current_len # 25-16=9
            startframe = random.randint(0, diffnum) # 6
            future_idx = startframe + current_len - 1 #6-1+16 = 21
        else:
            newLen = int(fnum * 2.0 / 3.0) # if number of .jpg files < 16; suppose fnum=10
            diffnum = fnum - newLen
            startframe = random.randint(0, diffnum)
            frame_gap = float(newLen - 1) / float(current_len)
            future_idx = int(startframe + current_len * frame_gap)


        crop_offset_x = -1
        crop_offset_y = -1
        ratio = random.random() * (4/3 - 3/4) + 3/4
        # reading video
        for i in range(self.videoLen): # videoLen=4

            nowid = int(startframe + i * frame_gap)             # 6+[0,1,2,3]*2 = [6,8,10,12]
            # img_path = folder_path + "{:02d}.jpg".format(nowid)
            # specialized for fouhey format
            newid = nowid + 1                                   # [7,9,11,13]
            img_path = folder_path + "{:06d}.jpg".format(newid) # [7.jpg, 9.jpg, 11.jpg, 13.jpg]

            img = load_image(img_path)  # CxHxW

            ht, wd = img.size(1), img.size(2)
            if ht <= wd:
                ratio  = float(wd) / float(ht)
                # width, height
                img = resize(img, int(self.imgSize * ratio), self.imgSize)
            else:
                ratio  = float(ht) / float(wd)
                # width, height
                img = resize(img, self.imgSize, int(self.imgSize * ratio))


            if crop_offset_x == -1:
                crop_offset_x = random.randint(0, img.size(2) - self.cropSize - 1) #imgSiz=256
                crop_offset_y = random.randint(0, img.size(1) - self.cropSize - 1)

            img = self.cropimg(img, crop_offset_x, crop_offset_y, self.cropSize)
            assert(img.size(1) == self.cropSize)
            assert(img.size(2) == self.cropSize)

            if self.is_train:
                # Flip
                if toflip:
                    img = torch.from_numpy(fliplr(img.numpy())).float()

            mean=[0.485, 0.456, 0.406]
            std=[0.229, 0.224, 0.225]
            img = color_normalize(img, mean, std)

            imgs[i] = img.clone()        # imgs: torch tensor:[4,3,240,240]

        # reading img
        # img_path = folder_path + "{:02d}.jpg".format(future_idx)
        # specialized for fouhey format

        for i in range(2):

            newid = int(future_idx + 1 + i * frame_gap) # int(21+1+i*2) --> [22, 24]
            if newid > fnum:                            # suppose: fnum=25
                newid = fnum                            # let newid=last frame
            img_path = folder_path + "{:06d}.jpg".format(newid)

            img = load_image(img_path)  # CxHxW
            ht, wd = img.size(1), img.size(2)
            newh, neww = ht, wd
            if ht <= wd:
                ratio  = float(wd) / float(ht)
                # width, height
                img = resize(img, int(self.imgSize * ratio), self.imgSize)
                newh = self.imgSize
                neww = int(self.imgSize * ratio)
            else:
                ratio  = float(ht) / float(wd)
                # width, height
                img = resize(img, self.imgSize, int(self.imgSize * ratio))
                newh = int(self.imgSize * ratio)
                neww = self.imgSize

            img = self.cropimg(img, crop_offset_x, crop_offset_y, self.cropSize)
            assert(img.size(1) == self.cropSize)
            assert(img.size(2) == self.cropSize)

            if self.is_train:
                if toflip:
                    img = torch.from_numpy(fliplr(img.numpy())).float()

            mean=[0.485, 0.456, 0.406]
            std=[0.229, 0.224, 0.225]
            img = color_normalize(img, mean, std)

            future_imgs[i] = img            # imgs: torch tensor:[4,3,240,240]

        for i in range(2):
            imgs_target[i] = future_imgs[i].clone() # [22, 24] frames: torch tensor:[4,3,240,240]


        flow_cmb = future_imgs[0] - future_imgs[1]  # frame22 - frame24
        flow_cmb = im_to_numpy(flow_cmb)
        flow_cmb = flow_cmb.astype(np.float)
        flow_cmb = np.abs(flow_cmb)                 # abs(frame22 - frame24); shape [3,240,240]

        side_edge = self.cropSize  # 240
        box_edge  = int(side_edge / self.gridSize)  # gridSize=9; box_edge=int(240/9)=26

        lblxset = []
        lblyset = []
        scores  = []
        # Get some/49 patches in that flow_cmb=future_imgs[0] - future_imgs[1]
        for i in range(self.gridSize - 2): # [0,...,6] 7 numbers
            for j in range(self.gridSize - 2): # [0,...,6] 7 numbers

                offset_x1 = i * box_edge 
                offset_y1 = j * box_edge
                lblxset.append(i) # [0, ... , 6]
                lblyset.append(j) # [0, ... , 6]
                # flow_cmb is a numpy array with shape: [240, 240, 3]
                # tpatch is a temp numpy array with shape: [78, 78, 3]
                tpatch = flow_cmb[offset_y1: offset_y1 + box_edge * 3, offset_x1: offset_x1 + box_edge * 3].copy() #tpatch is one patch in 3x3 patches
                tsum = tpatch.sum() # pixel intensity sum over [78, 78, 3]
                scores.append(tsum)


        scores = np.array(scores)
        ids = np.argsort(scores) # Returns the indices that would sort an array.
        ids = ids[-10: ]         # find out the patches which are not blank! Top10 largest pixel value patches
        lbl = random.randint(0, 9) # suppose 6
        lbl = ids[lbl]            # ids[6]: the 4th large pixel patch

        lbl_x = lblxset[lbl] # suppose is 2
        lbl_y = lblyset[lbl] # suppose is 2


        if self.is_train:
            if toflip:
                lbl_x = self.gridSize - 3 - lbl_x

        lbl   = lbl_x * (self.gridSize - 2) + lbl_y # 2*7+2 = 16 somehow not used later!

        xloc = lbl_x / 6.0 # 0.333
        yloc = lbl_y / 6.0 # 0.333

        theta_aff = np.random.rand(6) # random number array with shape (6, )
        scale = 1.0 - 1.0 / 3.0       # 0.666
        randnum = (np.random.rand(2) - 0.5) / 6.0 # np.array with shape (2, )
        xloc = xloc + randnum[0] # say (2-0.5+0.7)/6=0.37
        yloc = yloc + randnum[1] # say (2-0.5+0.4)/6=0.32

        if xloc < 0:
            xloc = 0.0
        if xloc > 1:
            xloc = 1.0

        if yloc < 0:
            yloc = 0.0
        if yloc > 1:
            yloc = 1.0
        # both xloc, yloc are numbers in the range [0, 1]
        # [-45, 45]
        alpha = (np.random.rand(1)-0.5)*2*np.pi*(1.0/4.0)
        # theta_aff is a list with shape
        theta_aff[2] = (xloc * 2.0 - 1.0) * scale
        theta_aff[5] = (yloc * 2.0 - 1.0) * scale
        theta_aff[0] = 1.0 / 3.0 *np.cos(alpha)
        theta_aff[1] = 1.0 / 3.0 *(-np.sin(alpha))
        theta_aff[3] = 1.0 / 3.0 *np.sin(alpha)
        theta_aff[4] = 1.0 / 3.0 *np.cos(alpha)

        theta = torch.Tensor(theta_aff.astype(np.float32))
        theta = theta.view(1, 2, 3) # torch tensor with shape [1, 2, 3]
        theta_batch = theta.repeat(2, 1, 1) # torch tensor with shape [2, 2, 3]
        patch_target = self.geometricTnf(image_batch=imgs_target, theta_batch=theta_batch)
        # patch target shape [2, 3, 80, 80]
        theta = theta.view(2, 3)

        imgs_target = imgs_target[0:1]   # [3, 240, 240]
        patch_target = patch_target[0:1] # [3, 240, 240]


        meta = {'folder_path': folder_path, 'startframe': startframe, 'future_idx': future_idx, 'frame_gap': float(frame_gap), 'crop_offset_x': crop_offset_x, 'crop_offset_y': crop_offset_y, 'dataset': 'vlog'}

        # imgs: [4, 3, 240, 240]
        # imgs_target: [3, 240, 240]
        # patch_target: [3, 80, 80]
        # theta: [2, 3]
        return imgs, imgs_target, patch_target, theta, meta

    def __len__(self):
        return len(self.jpgfiles) # the number of scans in our case
