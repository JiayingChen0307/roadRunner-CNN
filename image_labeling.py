import numpy as np
import random
import torch
import csv
import shutil,os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch.utils.data import Dataset, DataLoader
from matplotlib.image import imread
from sklearn.datasets.samples_generator import (make_blobs,
                                                make_circles,
                                                make_moons)
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score

from Tina.dataset import Satellite_Dataset

dataset = Satellite_Dataset("log.csv")
category0,category1 = dataset.get_pass_fail_list()

#ploting z_scores 
z_scores = dataset.get_z_scores()
z_scores = [float(z) for z in z_scores]

plt.figure(figsize=(6, 6))
plt.scatter(z_scores, [1]*len(z_scores))
plt.xlabel('z scores')
plt.ylabel('1')
plt.title('Visualization of z-scores')

plt.hist(z_scores)
sum(z_scores)/len(z_scores)

#random.shuffle(road_0)
#random.shuffle(road_1)
#n_c0_02 = int(len(road_0)*0.2)
#n_c1_02 = int(len(road_1)*0.2)
#for f in road_0[:n_c0_02*3]:
#   shutil.copy(f,'Tina/labeled_png/train/category0')    
#for f in road_0[n_c0_02*3:n_c0_02*4]:
#   shutil.copy(f,'Tina/labeled_png/val/category0')
#for f in road_0[n_c0_02*4:]:
#   shutil.copy(f,'Tina/labeled_png/test/category0')
#for f in road_1[:n_c1_02*3]:
#   shutil.copy(f,'Tina/labeled_png/train/category1')
#for f in road_1[n_c1_02*3:n_c1_02*4]:
#   shutil.copy(f,'Tina/labeled_png/val/category1')
#for f in road_1[n_c1_02*4:]:
#   shutil.copy(f,'Tina/labeled_png/test/category1')
#print('train_c0:',n_c0_02*3)
#print('train_c1:',n_c1_02*3)
#print('val_c0/test_c0:',n_c0_02)
#print('val_c1/test_c1:',n_c1_02)

# this function puts png files into corresponding folders
# parameters: 
# category0: the list of image file names in category 0 
# category1: the list of image file names in category 1
# percent: the unit used to decide the size of train/test/val datasets
# directory: the directory to store the images
def image_labeling (category0,category1,directory):
    #rebuild data storage structures
    if os.path.isdir('Tina/labeled_png'):
        shutil.rmtree('Tina/labeled_png')
        os.mkdir('Tina/labeled_png')
        list_dirs = ['Tina/labeled_png/train/category0',
                 'Tina/labeled_png/train/category1',
                 'Tina/labeled_png/val/category0',
                 'Tina/labeled_png/val/category1',
                 'Tina/labeled_png/test/category0',
                 'Tina/labeled_png/test/category1']
        for d in list_dirs:
            if not(os.path.isdir(d)):
               os.makedirs(d)

    category = [category0,category1]
    cate_str = ['category0','category1']
    # train 60%; val 20%; test 20%
    for c in category:
        random.shuffle(c)    
        num = int(len(c)*0.2)
        stage = {'start':0,'train':num*3,'val':num*4,'test':num*5}
        key_list = list(stage.keys())
        for k in key_list[1:]:
            path = os.path.join(*[directory,k,cate_str[category.index(c)]]) 
            start = stage[key_list[key_list.index(k)-1]]
            end = stage[k]
            for f in c[start:end]:                      
               shutil.copy(f,path)   
                
image_labeling(category0,category1,'Tina/labeled_png') 

print('train_c0:',len(os.listdir('Tina/labeled_png/train/category0')))
print('train_c1:',len(os.listdir('Tina/labeled_png/train/category1')))
print('val_c0/test_c0:',len(os.listdir('Tina/labeled_png/val/category0')))
print('val_c1/test_c1:',len(os.listdir('Tina/labeled_png/val/category1')))
print('image_total:',len(os.listdir('processing/png')))