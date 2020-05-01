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


# modified Calvin's 'Satellite_Dataset'
class Satellite_Dataset(Dataset):
  

  # eventually, make this a list of lists to include id and iriz
  def __init__(self, csv_filepath,labeling_method='default',acceptable_iriz_roughness=0.04):
    
    satellite_image_path_prefix = "processing/png/"
    satellite_image_path_suffix = "_8.png"
    
    self.image_filenames = []
    self.iri_z_scores = []
    self.method = labeling_method
    self.acceptable_iriz_roughness = acceptable_iriz_roughness
    
    with open(csv_filepath) as log_file:
      
      csv_reader = csv.reader(log_file)
      
      count = 0
      for row in csv_reader:
        #print(row)
        
        count += 1
        new_entry = row[9]
        new_entry = satellite_image_path_prefix + new_entry
        new_entry = new_entry + satellite_image_path_suffix
        
        self.image_filenames.append(new_entry)
        
        
        self.iri_z_scores.append(row[8])
        
        
      self.image_filenames.remove(satellite_image_path_prefix + "filename" + satellite_image_path_suffix)
      self.iri_z_scores.remove("iriZ")
      
      
    if self.method == 'default':
      self.__default_threshold__()
    elif self.method == 'kmeans':
      self.__kmeans_threshold__() 
    
    #balance the dataset
    self.__balance_dataset()
    pass_,fail = self.get_percent_pass_fail()
    
    #label the data again
    if self.method == 'default':
      self.__default_threshold__()
      
  def __len__(self):
    return len(self.image_filenames)
    
    
  def view_image(self, image_index):
        
    #print(self.image_filenames)
    plt.imshow(mpimg.imread(self.image_filenames[image_index]))
  
  def get_z_scores(self):
    return self.iri_z_scores
  
  def get_image_filenames(self):
    return self.image_filenames
  
  #return two lists(categories) of image filenames
  def __kmeans_threshold__(self):
    d2_zscores = []
    for z in self.iri_z_scores:
        pair = [None]*2
        pair[0] = z
        pair[1] = 1
        d2_zscores.append(pair)

    X = np.array(d2_zscores)
    km = KMeans(n_clusters=2, random_state=0)
    km.fit(X)
    plt.scatter(X[km.labels_ == 0, 0], X[km.labels_ == 0, 1],
                      c='green', label='cluster 1')
    plt.scatter(X[km.labels_ == 1, 0], X[km.labels_ == 1, 1],
                      c='blue', label='cluster 2')

    self.i_0_kmeans, = np.where(km.labels_ == 0)
    self.road_0_kmeans = [self.image_filenames[index] for index in self.i_0_kmeans]
    self.i_1_kmeans, = np.where(km.labels_ == 1)
    self.road_1_kmeans = [self.image_filenames[index] for index in self.i_1_kmeans]
  
  #helper method for get_pass_fail and get_pass_fail_list
  def __default_threshold__(self):
    
    self.i_0 = []
    self.i_1 = []
    
    for index in range(len(self.iri_z_scores)):
      if (float(self.iri_z_scores[index]) < self.acceptable_iriz_roughness):
        self.i_0.append(index)
      else:
        self.i_1.append(index)
        
    self.road_0 = [self.image_filenames[index] for index in self.i_0]
    self.road_1 = [self.image_filenames[index] for index in self.i_1]
  
  #return a list of normal roads and a list of bumpy roads
  def get_pass_fail_list(self):
     
    if self.method == 'default':
      return self.road_0, self.road_1
    
    elif self.method == 'kmeans':   
      return self.road_0_kmeans, self.road_1_kmeans 
  
  #tells whether a road is normal or bumpy by index
  # 0 is normal, 1 is bumpy
  def get_pass_fail(self,index):   
    
    if self.method == 'default':   
      if index in self.i_0:
        return 0
      else:
        return 1   
  
    elif self.method == 'kmeans':
      if index in self.i_0_kmeans:
        return 0
      else:
        return 1
  
  #used for experiment recording
  #counts the percentage of normal roads and bumpy roads
  def get_percent_pass_fail(self):
    
    pass_ = 0
    total = len(self.iri_z_scores)
    for i in range(total):
      if self.get_pass_fail(i) == 0:
        pass_ += 1
    fail = total - pass_
  
    return pass_/total,fail/total

  # This method balances the dataset by duplicating datapoints until there is
  #     an equal number of passing and failing roads in the dataset.
  #credit to Calvin
  def __balance_dataset(self):
    
    self.good_road_data = []
    self.bad_road_data = []
    
    prev_len = len(self)
    
    for i in range (0, len(self.iri_z_scores) - 2):
      if (self.get_pass_fail(i) == 0):
        self.good_road_data.append(i)
      else:
        self.bad_road_data.append(i)
        
        
        
    if (len(self.bad_road_data) > len(self.good_road_data)):
      #while (len(self.bad_road_data) > len(self.good_road_data)):
        random.shuffle(self.good_road_data);
        
        x = 1
        
        for i in self.good_road_data:
          
          
          self.image_filenames.append(self.image_filenames[i])
        
        
          self.iri_z_scores.append(self.iri_z_scores[i])
        
          #self.dataset_id.append(x+ prev_len)
          
          #self.is_duplicate.append(i)
          
          #print("diff between more bad: ", len(self.good_road_data) - len(self.bad_road_data))


          
          self.good_road_data.append(x + prev_len)
          
          x += 1
          
          if (len(self.good_road_data) > len(self.bad_road_data)):
            break
          
          
          
    elif (len(self.good_road_data) > len(self.bad_road_data)):
      #while (len(self.good_road_data) > len(self.bad_road_data)):
        random.shuffle(self.bad_road_data);
        
        x  = 1
        
        for i in self.bad_road_data:
          
          
          self.image_filenames.append(self.image_filenames[i])
        
        
          self.iri_z_scores.append(self.iri_z_scores[i])
        
          #self.dataset_id.append(x + prev_len)
          
          #self.is_duplicate.append(i)
          
          #print("diff between: ", len(self.good_road_data) - len(self.bad_road_data))
          
          self.bad_road_data.append(x + prev_len)
          
          x += 1
    
    
          if (len(self.bad_road_data) > len(self.good_road_data)):
            break

  def __getitem__(self, index):
    
    tensor_image = self.tensor_image(index)
    
    return tensor_image, self.get_pass_fail(index)

  
#dataset = Satellite_Dataset("log.csv")
#dataset.get_pass_fail_list()