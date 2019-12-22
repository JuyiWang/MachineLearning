import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimz

import matplotlib.pyplot as plt
import numpy as np

from classfier import imshow,classes,cnn

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

testset = torchvision.datasets.CIFAR10(root = '/Users/Juyi/Desktop/Code/pytorch/data',
                                            train = False,download = True,transform = transform)
                                            
testLoader = torch.utils.data.DataLoader(testset,batch_size = 4,shuffle = True,num_workers = 2)

def ViewFinish():
    dataiter = iter(testLoader)
    images,labels = dataiter.next()

    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    net = cnn()
    net.load_state_dict(torch.load('/Users/Juyi/Desktop/Code/pytorch/model/cifar_net.pth'))

    output = net(images)
    #max(input,0||1) 0返回列最大值，1返回行最大值
    _,predicted = torch.max(output,1)
    print('GroundPredict: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

if __name__ == '__main__':
    cnn_test = cnn()
    cnn_test.load_state_dict(torch.load('/Users/Juyi/Desktop/Code/pytorch/model/cifar_net.pth'))
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testLoader:
            images,labels = data
            output = cnn_test(images)
            _,predict = torch.max(output,1)
            total += labels.size(0)
            correct += (predict == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))