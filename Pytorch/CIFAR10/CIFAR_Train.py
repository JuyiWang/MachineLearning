import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimz

import matplotlib.pyplot as plt
import numpy as np

import logging

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class cnn(nn.Module):
    def __init__(self):
        super(cnn,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64,128,5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.Relu1 = nn.Sequential(
            nn.Linear(128 * 5 * 5,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU()
        )
        self.Fc = nn.Linear(84,10)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        out = out.view(-1,128 * 5 * 5)
        out = self.Relu1(out)
        out = self.Fc(out)
        return out

def readData():
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    trainset = torchvision.datasets.CIFAR10(root = '/Users/Juyi/Desktop/Code/pytorch/data',
                                                train = True,download = True,transform = transform)
    trainLoader = torch.utils.data.DataLoader(trainset,batch_size = 4,shuffle = True,num_workers = 2)
    return trainset,trainLoader

# def imshow(img):
#     img = img /2 + 0.5
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg,(1,2,0)))
#     plt.show()

def train():
    trainset,trainLoader = readData()

    net = cnn()

    criterion = nn.CrossEntropyLoss()
    optimzer = optimz.Adam(net.parameters(),lr = 0.001)

    epoch_num = 10

    for epoch in range(epoch_num):
        for i,data in enumerate(trainLoader):
            inputs,lables = data

            optimzer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs,lables)
            loss.backward()

            optimzer.step()
            
            running_loss = loss.item()
            if i % 2000 == 1999:  
                print('[%d, %5d] loss: %.6f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print("Finished Training")
    PATH = '/Users/Juyi/Desktop/Code/pytorch/model/cifar_net.pth'
    torch.save(net.state_dict(),PATH)

if __name__ == '__main__':
    logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("Training is begin!")
    print(id(__name__))
    print(id('__name__'))
    train()

