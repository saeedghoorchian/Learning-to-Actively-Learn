import random
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms


transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                         std  = [ 0.229, 0.224, 0.225 ]),
    ])

# trainData = dsets.ImageFolder('../data/imagenet/train', transform)
# testData = dsets.ImageFolder('../data/imagenet/test', transform)

# trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
# testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=False)

def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        nn.BatchNorm2d(chann_out),
        nn.ReLU()
    )
    return layer

def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):

    layers = [ conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list)) ]
    layers += [ nn.MaxPool2d(kernel_size = pooling_k, stride = pooling_s)]
    return nn.Sequential(*layers)

def vgg_fc_layer(size_in, size_out):
    layer = nn.Sequential(
        nn.Linear(size_in, size_out),
        nn.BatchNorm1d(size_out),
        nn.ReLU()
    )
    return layer

class vgg(nn.Module):
    def __init__(self, output_shape_of_penultimate_layer, num_classes):
        super(vgg, self).__init__()

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)

        # FC layers
        self.layer6 = vgg_fc_layer(7*7*512, 4096)
        self.layer7 = vgg_fc_layer(4096, output_shape_of_penultimate_layer) #self.layer7 = vgg_fc_layer(4096, 4096)

        # Final layer
        self.layer8 = nn.Linear(output_shape_of_penultimate_layer, num_classes) #self.layer8 = nn.Linear(4096, num_classes)
        self.layer9 = nn.Softmax(dim = -1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        vgg16_features = self.layer5(out)
        out = vgg16_features.view(out.size(0), -1)
        out = self.layer6(out)
        penult_out = self.layer7(out)
        logits = self.layer8(penult_out)
        probs = self.layer9(logits) 

        return probs, logits, penult_out


class CNN(nn.Module):
    def __init__(self, output_shape_of_penultimate_layer, num_classes):
        super(CNN, self).__init__()
        self.convs = nn.Sequential(
                                nn.Conv2d(1,32,4),
                                nn.ReLU(),
                                nn.Conv2d(32,32,4),
                                nn.ReLU(),
                                nn.MaxPool2d(2),
                                nn.Dropout(0.25)
        )
#         self.fcs = nn.Sequential(
#                                 nn.Linear(11*11*32,128),
#                                 nn.ReLU(),
#                                 nn.Dropout(0.5),
#                                 nn.Linear(128,num_classes), #10
#         )

        # FC layers
        self.layer6 = nn.Linear(11*11*32,output_shape_of_penultimate_layer) #128
        self.layer7 = nn.ReLU()

        # Final layer
        self.layer8 = nn.Dropout(0.5)
        self.layer9 = nn.Linear(output_shape_of_penultimate_layer, num_classes) #self.layer8 = nn.Linear(4096, num_classes)
        self.layer10 = nn.Softmax(dim = -1)

        
    def forward(self, x):
        out = x
        out = self.convs(out)
        out = out.view(-1,11*11*32)
#         out = self.fcs(out)
        penult_out = self.layer6(out)
        out = self.layer7(penult_out)
        out = self.layer8(out)
        logits = self.layer9(out)
        probs = self.layer10(logits)
        return probs, logits, penult_out

