import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn.functional as Fun
import matplotlib.pyplot as plt

class normal_block(nn.Module):
  factor=1
  def __init__(self,inp,out,stride=1,change=None):
    super(normal_block,self).__init__()
    #step1
    self.con1=nn.Conv2d(inp,out,stride=stride,kernel_size=3,padding=1)
    self.norm1=nn.BatchNorm2d(out)
    self.con2=nn.Conv2d(out,out,stride=1,kernel_size=3,padding=1)
    self.norm2=nn.BatchNorm2d(out)
    self.change=change

  def forward(self,x):

    x1=x
    out=self.con1(x)
    out=self.norm1(out)
    out=Fun.relu(out)
    out=self.con2(out)
    out=self.norm2(out)

    if self.change is not None:
      x1=self.change(x1)
    out+=x1
    out=Fun.relu(out)

    return out

class side_block(nn.Module):
  factor=4
  def __init__(self,inp,out,stride=1,change=None):
    super(side_block,self).__init__()
    

    self.con1=nn.Conv2d(inp,out,kernel_size=1,stride=1)
    self.norm1=nn.BatchNorm2d(out)
    self.con2=nn.Conv2d(out,out,stride=stride,kernel_size=3,padding=1)
    self.norm2=nn.BatchNorm2d(out)
    self.con3=nn.Conv2d(out,out*self.factor,kernel_size=1)
   
    
    self.norm3=nn.BatchNorm2d(out*self.factor)
    self.change=change

  def forward(self,x):
    x1=x
    out=self.con1(x)
    out=self.norm1(out)
    out=Fun.relu(out)
    out=self.con2(out)
    out=self.norm2(out)
    out=Fun.relu(out)
    out=self.con3(out)
    out=self.norm3(out)
    if self.change is not None:
      x1=self.change(x1)
    out+=x1
    out=Fun.relu(out)

    return out


class resnet(nn.Module):
  def __init__(self,block,layer,classes=10):
    super(resnet,self).__init__()

    self.inp=64
    self.con1=nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1)
    self.norm1=nn.BatchNorm2d(64)
    self.step1=self._layer(block,64,layer[0],stride=1)
    self.step2=self._layer(block,128,layer[1],stride=2)
    self.step3=self._layer(block,256,layer[2],stride=2)
    self.step4=self._layer(block,512,layer[3],stride=2)

    self.pool=nn.AvgPool2d(kernel_size=4,stride=1)

    self.fc=nn.Linear(512*block.factor,classes)


  def _layer(self,block,out,layer,stride=1):
    change=None

    if stride!=1 or out!=self.inp*block.factor:
      change=nn.Sequential(nn.Conv2d(self.inp,out*block.factor,kernel_size=1,stride=stride),nn.BatchNorm2d(out*block.factor))

    final=[]

    final.append(block(self.inp,out,stride=stride,change=change))
    self.inp=out*block.factor

    for j in range(1,layer):
      final.append(block(self.inp,out))
      self.inp=out*block.factor

    return nn.Sequential(*final)

  def forward(self,x):
    x=self.con1(x)
    x=self.norm1(x)
    x=Fun.relu(x)
    x=self.step1(x)
    x=self.step2(x)
    x=self.step3(x)
    x=self.step4(x)

    x=Fun.avg_pool2d(x,4)
    x=x.view(x.size(0),-1)
    x=self.fc(x)


    return x


     
