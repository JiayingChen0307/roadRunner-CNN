# modified from Sasank Chilamkurthy's tutorial

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
from matplotlib import image

import pickle

from Tina.dataset import Satellite_Dataset
from Tina import data_subset
from Tina.tracker import Tracker

tracker = Tracker()
db = tracker.import_db('Tina/history')
model = db[0]
model['recall'] = model['bumpy_correct']/(model['bumpy_correct']
                                           +model['normal_total']-model['normal_correct'])
print(model)
record = {}

batch_size = 4
epoch = 25
threshold = 0.04
loss_fc_weights = [1.0,1.0]
plt.ion()   # interactive mode

#calculate mean and std for the training set
#mean,std = cal_m_std()
#print(mean,std)

#load data
dataset = Satellite_Dataset("log.csv",'default',threshold)
training_set = data_subset.Data_Subset("log.csv",'train',dataset)
mean,std = training_set.cal_mean_std()
image_datasets = {x:data_subset.Data_Subset("log.csv",x,dataset,mean,std) for x in ['val','test']}
image_datasets['train'] = training_set
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val','test']}
#class_names = image_datasets['train'].classes
#dataset_sizes = {'train':len(training_dataset),'val':len(validation_dataset)}
class_names = ['category0','category1']
#print(dataset_sizes)
#print(class_names)
#print(dataloaders)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, scheduler, num_epochs=epoch):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    train_acc = []
    val_acc = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    #print('pred is:',preds)
                    #print(outputs,labels)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                    
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            #ploting the train/val accuracy
            if phase == 'train':
                train_acc.append(float(epoch_acc))
            else:
                val_acc.append(float(epoch_acc))
          
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print(train_acc)
    print(val_acc)
    record['performance']='{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)
    record['best_acc']=best_acc
    
    #x = np.arange(0,num_epochs)
    #line1 = plt.plot(train_acc,label = 'train')
    #plt.plot(x,line1,'g+-',x,line2,'b^-')
    #plt.title('Model Evaluation')
    #plt.xlabel('epoch')
    #plt.ylabel('accuracy')
    #plt.legend()
    #plt.show()
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


model_conv = torchvision.models.resnet18(pretrained=True)
#model_conv = torchvision.models.vgg16(pretrained=True)
#model_conv = torchvision.models.vgg19_bn(pretrained=True)
#model_conv = torchvision.models.vgg16_bn(pretrained=True)
#model_conv = torchvision.models.resnet50(pretrained=True)
#model_conv = torchvision.models.densenet161(pretrained=True)
#inception = models.inception_v3(pretrained=True)
#googlenet = models.googlenet(pretrained=True)

#selected pretrained models to test

for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default    
    
##vgg    
#n_inputs = model_conv.classifier[6].in_features
#last_layer = nn.Linear(n_inputs, 2)
#model_conv.classifier[6] = last_layer

##densenet
#n_inputs = model_conv.classifier.in_features
#last_layer = nn.Linear(n_inputs, 2)
#model_conv.classifier = last_layer

##resnet
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

#criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss(weight=torch.tensor(loss_fc_weights))

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)

# Testing

test_loss = 0.0
#model_conv.eval()

class_correct = [0, 0]
class_total = [0, 0]
class_1_as_0 = 0
class_0_as_1 = 0

#def my_mse_loss(outputs,target):
#  _, preds = torch.max(outputs, 1)
#  preds = preds.type(torch.FloatTensor)
#  loss = nn.MSELoss(preds,target)
#  return loss

#test_image_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'),
#                                          data_transforms['test'])
#test_loader = torch.utils.data.DataLoader(test_image_dataset, batch_size=4,
#                                             shuffle=True)
for data, target in dataloaders['test']:
  output = model_conv(data)
  loss = criterion(output, target)
  test_loss += loss.item()*data.size(0)
  
  
  _, pred = torch.max(output, 1)
  
#  print("pred is: ",pred)
  
  correct_tensor = pred.eq(target.data.view_as(pred))
  correct = np.squeeze(correct_tensor.numpy())
  
  
 # print("len data: ",len(correct))
  
  for i in range(batch_size):
    
    if (len(pred) == batch_size):
      
      class_total[target[i]] += 1
        
      if (pred[i] == target[i]):
        class_correct[target[i]] += 1

      
    

    
    
test_loss = test_loss / len(dataloaders['test'].dataset)
test_acc = 100 * (class_correct[0] + class_correct[1]) / (class_total[0] + class_total[1])
scaled_test_acc = 50 * class_correct[0] / class_total[0] + 50 * class_correct[1] / class_total[1]

print("Test loss is: ",test_loss)

print("Test accuracy for good roads: ", test_acc)

print("Test accuracy for bad roads: ", 100 * class_correct[1] / class_total[1])

print("Test accuracy for all roads: ", test_acc)

print("Scaled accuracy for all roads: ", scaled_test_acc)

print("Recall: ", class_correct[0]/(class_correct[0]+class_total[1]-class_correct[1]))

print(class_correct)
print(class_total)

def save2history(record,model,note=None):

  record['batch'] = batch_size
  record['epoch'] = epoch
  record['threshold'] = threshold
  record['loss_fc_weights'] = loss_fc_weights
  record['training_size']=dataset_sizes['train']
  record['val_size']=dataset_sizes['val']
  record['percent_normal'],record['percent_bumpy']=dataset.get_percent_pass_fail()
  record['model_name']=model
  record['test_loss']=test_loss
  record['test_acc']=test_acc
  record['normal_correct']=class_correct[0]
  record['bumpy_correct']=class_correct[1]
  record['normal_total']=class_total[0]
  record['bumpy_total']=class_total[1]
  record['scaled_test_acc']=scaled_test_acc
  record['addtional_note']=note

  tracker.update_db(db,record)
  print('Experiment saved to history.')

save2history('resnet18')
