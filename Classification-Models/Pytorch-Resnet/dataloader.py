import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn.functional as Fun
import matplotlib.pyplot as plt
import argparse

def ddata():


 train=datasets.CIFAR10(
   root="data",
   train=True,
     download=True,
    transform=transforms.ToTensor()

  )

 test=datasets.CIFAR10(
    root="data",
    train=False,
    
    transform=transforms.ToTensor()

  )

 training_data=torch.utils.data.DataLoader(train,batch_size=128,shuffle=True)
 test_data=torch.utils.data.DataLoader(test,batch_size=128,shuffle=False)
 return training_data,test_data
