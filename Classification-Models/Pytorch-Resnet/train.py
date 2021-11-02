import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn.functional as Fun
import matplotlib.pyplot as plt
import argparse
from model import normal_block,resnet,side_block
from dataloader import ddata

###################################################################################
# Parsing all arguments

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type = float, 
	default = 0.01, help = 'learning rate for the model')
parser.add_argument('--n_epoch', type = int, 
	default = 25, help = 'number of epoch')
parser.add_argument('--batch_size', type = int, 
	default = 128, help = '# of batch size')
parser.add_argument('--momentum', type = float, 
	default = 0.9, help = 'momentum for optimization')
#####################################################################################


def train(args):

  losses=[]
  accuracy=[]
  training_data,test_data=ddata()
  device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(device)

  model=resnet(side_block,[3,4,6,3])
  model.to(device)

  cost=nn.CrossEntropyLoss()
  optimizer=torch.optim.SGD(model.parameters(),lr=args.learning_rate,momentum=args.momentum)

  for epoch in range(args.n_epoch):
    loss1=0
    for i,batch in enumerate(training_data,0):
   # print(batch[0])
      dt,out=batch
    #print(dt)
      dt=dt.to(device)
      out=out.to(device)
      pred=model(dt)
      loss=cost(pred,out)
      loss1=loss.item()

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if i%100 ==0:
        print("Epoch",epoch," ",i," ",loss1)
        losses.append(loss1)
        loss1=0
    correct=0
    total=0
    for m in test_data:
      data,expc=m
      data=data.to(device)
      expc=expc.to(device)
      pred=model(data)
      _,pred=torch.max(pred.data,1)
      total+=expc.size(0)
      correct+=(pred==expc).sum().item()
    print("Accuracy on epoch",epoch," ",correct/total)
    accuracy.append((correct/total)*100)
    return losses,accuracy


args = parser.parse_args()
losses,accuracy=train(args)


#Accuracy plot

plt.plot(accuracy)
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.show()


#Loss plot

plt.plot(losses)
plt.ylabel("Loss")
plt.show()
