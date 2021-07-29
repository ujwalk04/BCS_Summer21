import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
import argparse
import requests
import urllib.request
import scipy.ndimage
class Resnet50(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        
        # pretrained resnet50 model
        self.resnet = resnet50(pretrained=True)
        
        # defining layers in Resnet architecture
        self.features = nn.Sequential(self.resnet.conv1,
                                      self.resnet.bn1,
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                      self.resnet.layer1, 
                                      self.resnet.layer2, 
                                      self.resnet.layer3, 
                                      self.resnet.layer4)
        
        # average pooling layer
        self.avgpool = self.resnet.avgpool
        
        # fully connected layer 
        self.classifier = self.resnet.fc
        
        # gradient placeholder
        self.gradient = None
    # defining various function to handle gradients of last layer
    # hook for the gradients
    def activations_hook(self, grad):
        self.gradient = grad
    
    def get_gradient(self):
        return self.gradient
    
    def get_activations(self, x):
        return self.features(x)
    
    def forward(self, x):
        
       
        x = self.features(x)              # feature extraction
        h = x.register_hook(self.activations_hook)  # register for hooks which would store the gradient
        x = self.avgpool(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        
        return x

      

transform = transforms.Compose([transforms.Resize((224, 224)),  # necessary transformations applied to the input image to make it the same as input format of our model 
                                transforms.ToTensor(), 
                                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])

parser = argparse.ArgumentParser()
parser.add_argument('--img_src', 
    default = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/35-1315924315-g.jpg/149px-35-1315924315-g.jpg", help = 'img_src')
args, unknown = parser.parse_known_args()

urllib.request.urlretrieve(img_src, 'image')
img1 = Image.open("image")
img1 = im.resize((224,224))  # Resizing image into size of (224,224)
plt.imshow(img1)
img=transform(img1)  # Transforming the input image 
img.unsqueeze_(0)    # Inplace operation which inserts dimension of size 1 at first position of tensor

# init the resnet
resnet = Resnet50()
_ = resnet.eval()  # Evaluating 
pred = resnet(img) 

pred.argmax(dim=1) # class with largest predicted probability

# get the gradients of the output with respect to the parameters of the model
pred[:, 2].backward()

# get the gradients out of the model
gradients = resnet.get_gradient()

# pooling the gradients
pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

# get the activations of the last convolutional layer
activations = resnet.get_activations(img).detach()

# Weight the channels of the map by the corresponding pooled gradients
for i in range(512):
    activations[:, i, :, :] *= pooled_gradients[i]
    
# Average the channels of the activations
heatmap = torch.mean(activations, dim=1).squeeze()

# Applying ReLU 
heatmap = np.maximum(heatmap, 0)

# Normalize the heatmap
heatmap /= torch.max(heatmap)

# Plotting heatmap
plt.matshow(heatmap.squeeze())

# Matching the size of heatmap with the image size[(224,224)]
heatmap=scipy.ndimage.zoom(heatmap,(32,32),order=1) 

# Superimposing heatmap over the base image
plt.imshow(im,alpha=0.5)
plt.imshow(heatmap,cmap='jet',alpha=0.5)
