import torch
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random
import torchvision.transforms as transforms
import numpy
from os.path import dirname
from PIL import Image

import os
from torchvision import transforms
from matplotlib import image

from Tina.dataset import Satellite_Dataset

class Data_Subset(Dataset):
  def __init__(self,filepath,stage,dataset,mean=None,std=None):
    
    
    
    #super().__init__(filepath)
    
    self.image_filenames = []
    self.iri_z_scores = []
    
    self.dataset = dataset
    #calculate mean and std for the training set
    #mean,std = cal_m_std()
    #print(mean,std)
    
    category0,category1 = self.dataset.get_pass_fail_list()
    category = [category0,category1]
    cate_str = ['category0','category1']
    # train 60%; val 20%; test 20%
    for c in category:
        random.shuffle(c)    
        num = int(len(c)*0.2)
        stages = {'start':0,'train':num*3,'val':num*4,'test':num*5}
        key_list = list(stages.keys())
        start = stages[key_list[key_list.index(stage)-1]]
        end = stages[stage]
        for f in c[start:end]: 
          self.image_filenames.append(f)
          idx = self.dataset.image_filenames.index(f)      
          self.iri_z_scores.append(self.dataset.iri_z_scores[idx]) 
          
    if stage == 'train':
      self.mean,self.std = self.cal_mean_std()
      
    else:
      self.mean = std
      self.std = mean
    
    # normalization for validation
    data_transforms = {
      'train': transforms.Compose([
          transforms.RandomResizedCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize(self.mean, self.std)
      ]),
      'val': transforms.Compose([
          transforms.Resize(256),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          transforms.Normalize(mean, std)
      ]), 
      'test': transforms.Compose([
          transforms.RandomResizedCrop(224),
          transforms.ToTensor(),
          transforms.Normalize(mean, std)
      ]),
}

    self.transform = data_transforms[stage]
     
     
  def __len__(self):
    return len(self.image_filenames)
  
  def __getitem__(self, index):
    
    f = self.image_filenames[index]
    pic = Image.open(f)
    tensor_image = self.transform(pic)
    original_idx = self.dataset.image_filenames.index(f)
    
    return tensor_image, self.dataset.get_pass_fail(original_idx)
  
  def cal_mean_std(self):
    
    channels_all = [[],[],[]]
    
    for f in self.image_filenames:
      pic = image.imread(f)
      pix = numpy.array(pic)
      x = pix.shape[0]
      y = pix.shape[1]

      #get the x*y array for all three channels
      channels = [pix[:, :, 0], pix[:, :, 1], pix[:, :, 2]]
      for n in range(3):
        channel = numpy.reshape(channels[n],(x*y)).tolist()
        channels_all[n] = channels_all[n] + channel
    mean = [numpy.mean(numpy.asarray(c)) for c in channels_all]
    std = [numpy.std(numpy.asarray(c)) for c in channels_all]
    
    return mean,std
#a = Data_Subset("log.csv")