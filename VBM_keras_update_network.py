# This file creates and trains the convolutional neural network for determing road roughness
# It uses a binary model (eg. roads are pass/fail)

import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import pickle
from os.path import dirname
import numpy as np
#from tensorflow.keras.utils.vis_utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers, models
import pandas as pd
import pylab as pl
import tensorflow as tf

from Tina import Adam_lr_mult
img_dim = 224
initial_epochs = 15
fine_tuning_epochs = 35
learning_rate = 0.001
fine_tune_learning_rate = 0.00001
batch_size = 10
sample_size_per_class = 200
kaggle = False
fine_tune_at = 13

#######################################
#load data
train_ds, val_ds, classweights, test_ds, train_df, test_df = create_generators(img_dim, batch_size, sample_size_per_class, kaggle)
steps_per_epoch=len(train_df)//batch_size

#######################################
#model and training 
base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_dim, img_dim, 3))
base_model.trainable = False
#def generate_vgg16():
#x = base_model(inputs, training=False)
inputs = tf.keras.Input(shape=(img_dim, img_dim, 3))
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dense(3, activation='softmax')(x)

model = tf.keras.Model(base_model.input, x)

model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy, 
              optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
              metrics=['accuracy'])


#model = generate_vgg16()
base_model.summary()
model.summary()

history = model.fit_generator(train_ds,
                              steps_per_epoch=steps_per_epoch, 
                              epochs=initial_epochs, 
                              verbose=1, 
                              validation_data=val_ds, 
                              class_weight=classweights)

####### fine-tuning
#print("Number of layers in the base model: ", len(base_model.layers))
# Fine-tune from this layer onwards

# make base model trainable
model.trainable = True

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False
  
# compile complete model
model.compile(optimizer = tf.keras.optimizers.Adam(fine_tune_learning_rate),
              loss = tf.keras.losses.sparse_categorical_crossentropy,
              metrics = ['accuracy'])
base_model.summary()
model.summary()

total_epochs = initial_epochs + fine_tuning_epochs
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
history_fine = model.fit_generator(train_ds,steps_per_epoch=steps_per_epoch, 
                                   epochs=total_epochs, 
                                   initial_epoch=history.epoch[-1],
                                   verbose=1, 
                                   validation_data=val_ds, 
                                   class_weight=classweights,
                                   callbacks=[callback])

#plot learning curves
def plot_learning_curve(initial_epochs):
  # plot learning curve
  acc = history.history['accuracy'] + history_fine.history['accuracy']
  val_acc = history.history['accuracy'] + history_fine.history['val_accuracy']

  plt.figure(figsize=(8, 8))
  plt.subplot(2, 1, 1)
  plt.plot(acc, label='Training Accuracy')
  plt.plot(val_acc, label='Validation Accuracy')
  plt.ylim([0.5, 1])
  plt.plot([initial_epochs,initial_epochs],
            plt.ylim(), label='Start Fine Tuning')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')
  
  loss = history.history['loss'] + history_fine.history['loss']
  val_loss = history.history['val_loss'] + history_fine.history['val_loss']
  plt.subplot(2, 1, 2)
  plt.plot(loss, label='Training Loss')
  plt.plot(val_loss, label='Validation Loss')
  plt.ylim([0, 1.0])
  plt.plot([initial_epochs,initial_epochs],
           plt.ylim(), label='Start Fine Tuning')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  plt.xlabel('epoch')
  plt.show()

plot_learning_curve(initial_epochs)

#model.save("Tina/vgg16")
#model = tf.keras.models.load_model("Tina/vgg16")

def test_evaluate():
# test network and show final accuracy and loss
  test_eval = model.evaluate_generator(test_ds, verbose=0)
  print('Test accuracy:', test_eval[1])
  print('Test loss:', test_eval[0])


  # show confusion matrix
  from sklearn.metrics import confusion_matrix
  predictions = model.predict(test_ds)
  predictions = np.argmax(np.round(predictions),axis=1)
  test_y = list(test_df['int_label'])
  cm = confusion_matrix(test_y,predictions)
  print(cm)
  print(predictions)

test_evaluate()
#####predictions using data from test_log.csv
# write predictions to test_results.csv
predictions = model.predict(test_ds)
predictions = np.argmax(np.round(predictions),axis=1)

p=0
with open('Tina/kaggle_results.csv', 'w', newline='') as test_results:
    results_fieldnames = ['id', 'prediction']
    results_writer = csv.DictWriter(test_results, fieldnames=results_fieldnames)
    results_writer.writeheader()
    with open('/var/lib/cdsw/share/test_log.csv') as test_log:
      test_log_reader = csv.reader(test_log)
      next(test_log_reader)
      for row in test_log_reader:
        idnum = row[0]
        results_writer.writerow({'id': idnum, 'prediction': predictions[p]})
        p += 1
        

#pl.matshow(cm)    
# show random training images
lab_pos = list(train_ds[5][1])
plt.figure(figsize=(20, 20))
images = train_ds[5][0]
for i in range(9):
  plt.subplot(3, 3, i + 1)
  plt.imshow(images[i])
  plt.title(int(lab_pos[i]))
  plt.axis("off")
  
# show random correctly classified images
test_x = list(test_df['filename'])
correct = np.intersect1d(np.where(np.array(test_y)==1), np.where(predictions==1))
for i in range(9):
  rand_num = random.randint(0, len(correct)-1)
  rand_image = test_x[correct[rand_num]]
  rand_image = plt.imread(rand_image)
  label = test_y[correct[rand_num]]
  plt.subplot(3, 3, i + 1)
  plt.imshow(rand_image)
  plt.title(int(label))
  plt.axis("off")

