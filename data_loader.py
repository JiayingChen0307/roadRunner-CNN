from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd

#######################################
# load data
def create_generators(img_dim, batch_size, sample_size_per_class, kaggle):
  
    train_filepath = "/var/lib/cdsw/share/train_log.csv"
    
    df = pd.read_csv(train_filepath)
    
    df_classes = [
      df[df['int_label'] == 0],
      df[df['int_label'] == 1],
      df[df['int_label'] == 2]]

    for idx in range(3):
      msk = np.random.rand(len(df_classes[idx]))< sample_size_per_class/len(df_classes[idx])
      df_classes[idx] = df_classes[idx][msk]

    df = pd.concat(df_classes, keys=['id','filename','CQV','prob_label','hot_label','int_label'])

    # calculate class weights
    classweights = {0:1., 
                    1:len(df_classes[0])/len(df_classes[1]), 
                    2:len(df_classes[0])/len(df_classes[2])}
  
    # split into training/val
    if kaggle:
      msk = np.random.rand(len(df))<0.75  
      train_df = df[msk]
      val_df = df[~msk]
      
      test_filepath = "/var/lib/cdsw/share/test_log.csv"
    
      test_df = pd.read_csv(test_filepath)
      int_label= [0] * len(test_df)
      test_df['int_label'] = int_label
      
    else:
      msk = np.random.rand(len(df))<0.6
      train_df = df[msk]
      val_test_df = df[~msk]
      msk = np.random.rand(len(val_test_df))<0.5
      val_df = val_test_df[msk]
      test_df = val_test_df[~msk]


    #train_val_df.groupby("quality").count()

    # create training dataset with rotation augmentation
    train_ds = ImageDataGenerator(rescale=1./255, rotation_range=360).flow_from_dataframe(
      dataframe=train_df,
      directory=None,
      x_col="filename",
      y_col = "int_label",
      class_mode = 'raw',
      target_size = (img_dim,img_dim),
      batch_size = batch_size,
    )

    # create val dataset
    val_ds = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
      dataframe=val_df,
      directory=None,
      x_col="filename",
      y_col = "int_label",
      class_mode = 'raw',
      target_size = (img_dim,img_dim),
      batch_size = batch_size,
    )

    # create test dataset
    test_ds = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
      dataframe=test_df,
      directory=None,
      x_col="filename",
      y_col = "int_label",
      class_mode = 'raw',
      target_size = (img_dim,img_dim),
      batch_size = batch_size,
      shuffle=False,
    )

    return train_ds, val_ds, classweights, test_ds, train_df, test_df