# Importing libraries
import numpy as np
import argparse
import torch
import torch.nn as nn
import requests, io
from torchvision.models import resnet50
import urllib.request
import scipy.ndimage
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils import data
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# Necessary transformations for our image
transform = transforms.Compose([transforms.Resize((224, 224)),  # necessary transformations applied to the input image to make it the same as input format of our model 
                                transforms.ToTensor(), 
                                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])


# Parsing values
parser = argparse.ArgumentParser()
parser.add_argument('--img_src', 
     default = 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/Grosser_Panda.JPG/1024px-Grosser_Panda.JPG', help = 'img src')
parser.add_argument('--eps', type = float,
     default = 0.5, help = 'epsilon')
args, unknown = parser.parse_known_args()

# Function to display test image along with its perturbed form
def display(img1,img2,label1,label2):
  fig = plt.figure(figsize=(10, 7))
  fig.add_subplot(1,2,1)
  plt.imshow(img1)
  plt.title(label1)
  fig.add_subplot(1,2,2)
  plt.imshow(img2)
  plt.title(label2)

# Predicts label of the image (Labels based on imagenet dataset extracted from "https://savan77.github.io/blog/labels.json"
def label_pred(tensor):
  labels_link = "https://savan77.github.io/blog/labels.json"    
  labels_json = requests.get(labels_link).json()
  labels = {int(idx):label for idx, label in labels_json.items()}
  idx=tensor.item()
  pred = labels[idx]
  return pred


# Retrieving Test Image 
urllib.request.urlretrieve(args.img_src, 'image')
img1 = Image.open("image")
img1 = img1.resize((224,224))  # Resizing image into size of (224,224)
plt.imshow(img1)
img=transform(img1)  # transforming images
img.unsqueeze_(0);  # Inplace operation which inserts dimension of size 1 at first position of tensor

# Resnet50 Model
resnet=resnet50(pretrained=True)

from torch.autograd import Variable
t_pred=388    #Label for giant panda class
t_pred = Variable(torch.LongTensor([t_pred]), requires_grad=False)

# Function which takes in the test image , perform operations to make it perturbed and then finds the wrongly predicted class
def fgsm(img, eps):
  img.requires_grad=True
  resnet.eval()
  pred = resnet(img) 
  tens=pred.argmax(dim=1)
  true_pred=label_pred(tens)[0:18]
  resnet.zero_grad()
  cost = loss(pred, t_pred)
  cost.backward()  
  attack = img + eps*img.grad.sign()
  attack = torch.clamp(attack, 0, 1)
  
  attack1=attack.squeeze(0)
  attack1=attack1[-1, :, :]
  attack1=attack1.detach().numpy()
  pred2=resnet(attack)
  tensor=pred2.argmax(dim=1)
  false_pred=label_pred(tensor)
  display(img1,attack1,true_pred,false_pred)


# Executing the function
eps=args.eps
fgsm( img,eps ) 





