import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F

class VanillaNet(nn.Module):
    def __init__(self, hidden_dim, out_dim, device, dtype):
        super(VanillaNet, self).__init__()
        self.hidden_dim1 = hidden_dim
        self.out_dim = out_dim
        self.device = device
        self.dtype = dtype

        self.fc1 = nn.Linear(1280, hidden_dim, dtype=dtype, device=device)
        self.fc2 = nn.Linear(hidden_dim, out_dim, dtype=dtype, device=device)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#clean
class LayerNormNet(nn.Module):
    def __init__(self, hidden_dim, out_dim, device, dtype, drop_out=0.1):
        super(LayerNormNet, self).__init__()
        self.hidden_dim1 = hidden_dim
        self.out_dim = out_dim
        self.drop_out = drop_out
        self.device = device
        self.dtype = dtype

        self.fc1 = nn.Linear(1280, hidden_dim, dtype=dtype, device=device)
        self.ln1 = nn.LayerNorm(hidden_dim, dtype=dtype, device=device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim,
                             dtype=dtype, device=device)
        self.ln2 = nn.LayerNorm(hidden_dim, dtype=dtype, device=device)
        self.fc3 = nn.Linear(hidden_dim, out_dim, dtype=dtype, device=device)
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        #print(x.shape)
        x = self.dropout(self.ln1(self.fc1(x)))
        x = torch.relu(x)
        x = self.dropout(self.ln2(self.fc2(x)))
        x = torch.relu(x)
        x = self.fc3(x)
        return x

class CNN1(nn.Module):

    def __init__(self, hidden_dim, out_dim, device, dtype, drop_out=0.1, kernel_size=3, stride=1, pool_size=3, pool_stride= 3, padding=0):
        super(CNN1, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 5, kernel_size, stride = stride, padding = padding, dtype=dtype, device=device),
            nn.BatchNorm1d(5, dtype=dtype, device=device),
            nn.ReLU(),
            nn.MaxPool1d(pool_size, stride = pool_stride))
        
        out1_size = int((1280 + 2*padding - (kernel_size - 1) - 1)/stride + 1)
        maxpool1_size = int((out1_size + 2*padding - (pool_size - 1) - 1)/pool_stride + 1)
        print ('maxpool1_size1', maxpool1_size*5)
        
        
        self.layer2 = nn.Sequential(
            nn.Conv1d(5, 5, kernel_size, stride = stride, padding = padding, dtype=dtype, device=device),
            nn.BatchNorm1d(5, dtype=dtype, device=device),
            nn.ReLU(),
            nn.MaxPool1d(pool_size, stride = pool_stride))
        out2_size = int((maxpool1_size + 2*padding - (kernel_size - 1) - 1)/stride + 1)
        maxpool2_size = int((out2_size + 2*padding - (pool_size - 1) - 1)/pool_stride + 1)
        print ('maxpool2_size', maxpool2_size*5)
        
        self.layer3 = nn.Sequential(
            nn.Conv1d(5, 5, kernel_size, stride = stride, padding = padding, dtype=dtype, device=device),
            nn.BatchNorm1d(5, dtype=dtype, device=device),
            nn.ReLU(),
            nn.MaxPool1d(pool_size, stride = pool_stride))
        out3_size = int((maxpool2_size + 2*padding - (kernel_size - 1) - 1)/stride + 1)
        maxpool3_size = int((out3_size + 2*padding - (pool_size - 1) - 1)/pool_stride + 1)
        print ('maxpool3_size', maxpool3_size*5)
        
        self.flatten = nn.Flatten()
        
        '''
        self.fc1 = nn.Linear(maxpool2_size*36, hidden_dim, dtype=dtype, device=device)
        self.ln1 = nn.LayerNorm(hidden_dim, dtype=dtype, device=device)
        self.relu = nn.ReLU()
        '''
        self.fc2 = nn.Linear(maxpool3_size*5, out_dim, dtype=dtype, device=device)
        self.dropout = nn.Dropout(p=drop_out)
        
    def forward(self, input):
        input = input.reshape(-1,1,1280)
        out = self.layer1(input)  
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.dropout(self.flatten(out))
        
        #out = self.dropout(self.ln1(self.fc1(out)))
        #out = self.relu(out)

        out = self.fc2(out)
        return out

class CNN2(nn.Module):

    def __init__(self, hidden_dim, out_dim, device, dtype, drop_out=0.2, kernel_size=3, stride=1, pool_size=3, pool_stride= 3, padding=0):
        super(CNN2, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 2, kernel_size, stride = stride, padding = padding, dtype=dtype, device=device),
            nn.BatchNorm1d(2, dtype=dtype, device=device),
            nn.ReLU(),
            nn.MaxPool1d(pool_size, stride = pool_stride))       
        out1_size = int((1280 + 2*padding - (kernel_size - 1) - 1)/stride + 1)
        maxpool1_size = int((out1_size + 2*padding - (pool_size - 1) - 1)/pool_stride + 1)
        print ('maxpool1_size', maxpool1_size*2)
        
        self.layer2 = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size, stride = stride, padding = padding, dtype=dtype, device=device),
            nn.BatchNorm1d(4, dtype=dtype, device=device),
            nn.ReLU(),
            nn.MaxPool1d(pool_size, stride = pool_stride))       
        out2_size = int((1280 + 2*padding - (kernel_size - 1) - 1)/stride + 1)
        maxpool2_size = int((out2_size + 2*padding - (pool_size - 1) - 1)/pool_stride + 1)
        print ('maxpool2_size', maxpool2_size*4)
        
        self.flatten = nn.Flatten()
        
        #self.drop1 = nn.Dropout(p=0.25)
        #self.fc1 = nn.Linear(maxpool1_size*2 + maxpool2_size*4, hidden_dim, dtype=dtype, device=device)
        #self.drop2 = nn.Dropout(p=0.25)
        #self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(maxpool1_size*2 + maxpool2_size*4, out_dim, dtype=dtype, device=device)        
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, input):

        input = input.reshape(-1,1,1280)
        
        out1 = self.layer1(input)  
        out1 = self.flatten(out1)
        
        out2 = self.layer2(input)
        out2 = self.flatten(out2)
        
        out = torch.cat([out1, out2], 1)
        out = self.dropout(out)
        
        #out = self.drop1(out)
        #out = self.fc1(out)
        #out = self.drop2(out)
        #out = self.relu1(out)
        out = self.fc2(out)
        
        return out

#d=0.15   0.05  3317；0.25  0.35
class CNN3(nn.Module):

    def __init__(self, hidden_dim, out_dim, device, dtype, drop_out=0.15, kernel_size=3, stride=1, pool_size=3, pool_stride= 3, padding=0):
        super(CNN3, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 2, kernel_size, stride = stride, padding = padding, dtype=dtype, device=device),
            nn.BatchNorm1d(2, dtype=dtype, device=device),
            nn.ReLU(),
            nn.MaxPool1d(pool_size, stride = pool_stride))       
        out1_size = int((1280 + 2*padding - (kernel_size - 1) - 1)/stride + 1)
        maxpool1_size = int((out1_size + 2*padding - (pool_size - 1) - 1)/pool_stride + 1)
        print ('maxpool1_size', maxpool1_size*2)
        
        self.layer2 = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size, stride = stride, padding = padding, dtype=dtype, device=device),
            nn.BatchNorm1d(4, dtype=dtype, device=device),
            nn.ReLU(),
            nn.MaxPool1d(pool_size, stride = pool_stride))       
        out2_size = int((1280 + 2*padding - (kernel_size - 1) - 1)/stride + 1)
        maxpool2_size = int((out2_size + 2*padding - (pool_size - 1) - 1)/pool_stride + 1)
        print ('maxpool2_size', maxpool2_size*4)
        
        self.layer3 = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size, stride = stride, padding = padding, dtype=dtype, device=device),
            nn.BatchNorm1d(4, dtype=dtype, device=device),
            nn.ReLU(),
            nn.MaxPool1d(pool_size, stride = pool_stride))       
        out3_size = int((1280 + 2*padding - (kernel_size - 1) - 1)/stride + 1)
        maxpool3_size = int((out3_size + 2*padding - (pool_size - 1) - 1)/pool_stride + 1)
        print ('maxpool3_size', maxpool3_size*4)
        
        self.flatten = nn.Flatten()
        
        #self.drop1 = nn.Dropout(p=0.25)
        #self.fc1 = nn.Linear(maxpool1_size*2 + maxpool2_size*4, hidden_dim, dtype=dtype, device=device)
        #self.drop2 = nn.Dropout(p=0.25)
        #self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(maxpool1_size*2 + maxpool2_size*4 + maxpool3_size*4, out_dim, dtype=dtype, device=device)        
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, input):

        input = input.reshape(-1,1,1280)
        
        out1 = self.layer1(input)  
        out1 = self.flatten(out1)
        
        out2 = self.layer2(input)
        out2 = self.flatten(out2)
        
        out3 = self.layer3(input)
        out3 = self.flatten(out3)
        
        out = torch.cat([out1, out2, out3], 1)
        out = self.dropout(out)
        
        #out = self.drop1(out)
        #out = self.fc1(out)
        #out = self.drop2(out)
        #out = self.relu1(out)
        out = self.fc2(out)
        
        return out
'''   
    
class CNN3(nn.Module):

    def __init__(self, hidden_dim, out_dim, device, dtype, drop_out=0.15, kernel_size=3, stride=1, pool_size=3, pool_stride= 3, padding=0):
        super(CNN3, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 2, 1, stride = stride, padding = padding, dtype=dtype, device=device),
            nn.BatchNorm1d(2, dtype=dtype, device=device),
            nn.ReLU(),
            nn.MaxPool1d(pool_size, stride = pool_stride))       
        out1_size = int((1280 + 2*padding - (1 - 1) - 1)/stride + 1)
        maxpool1_size = int((out1_size + 2*padding - (pool_size - 1) - 1)/pool_stride + 1)
        print ('maxpool1_size', maxpool1_size*2)
        
        self.layer2 = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size, stride = stride, padding = padding, dtype=dtype, device=device),
            nn.BatchNorm1d(4, dtype=dtype, device=device),
            nn.ReLU(),
            nn.MaxPool1d(pool_size, stride = pool_stride))       
        out2_size = int((1280 + 2*padding - (kernel_size - 1) - 1)/stride + 1)
        maxpool2_size = int((out2_size + 2*padding - (pool_size - 1) - 1)/pool_stride + 1)
        print ('maxpool2_size', maxpool2_size*4)
        
        self.layer3 = nn.Sequential(
            nn.Conv1d(1, 4, 5, stride = stride, padding = padding, dtype=dtype, device=device),
            nn.BatchNorm1d(4, dtype=dtype, device=device),
            nn.ReLU(),
            nn.MaxPool1d(pool_size, stride = pool_stride))       
        out3_size = int((1280 + 2*padding - (5 - 1) - 1)/stride + 1)
        maxpool3_size = int((out3_size + 2*padding - (pool_size - 1) - 1)/pool_stride + 1)
        print ('maxpool3_size', maxpool3_size*4)
        
        self.flatten = nn.Flatten()
        
        #self.drop1 = nn.Dropout(p=0.25)
        #self.fc1 = nn.Linear(maxpool1_size*2 + maxpool2_size*4, hidden_dim, dtype=dtype, device=device)
        #self.drop2 = nn.Dropout(p=0.25)
        #self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(maxpool1_size*2 + maxpool2_size*4 + maxpool3_size*4, out_dim, dtype=dtype, device=device)        
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, input):

        input = input.reshape(-1,1,1280)
        
        out1 = self.layer1(input)  
        out1 = self.flatten(out1)
        
        out2 = self.layer2(input)
        out2 = self.flatten(out2)
        
        out3 = self.layer3(input)
        out3 = self.flatten(out3)
        
        out = torch.cat([out1, out2, out3], 1)
        out = self.dropout(out)
        
        #out = self.drop1(out)
        #out = self.fc1(out)
        #out = self.drop2(out)
        #out = self.relu1(out)
        out = self.fc2(out)
        
        return out
        
'''
'''   
class CNN4(nn.Module):

    def __init__(self, hidden_dim, out_dim, device, dtype, drop_out=0.15, kernel_size=3, stride=1, pool_size=3, pool_stride= 3, padding=0):
        super(CNN4, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 2, kernel_size, stride = stride, padding = padding, dtype=dtype, device=device),
            nn.BatchNorm1d(2, dtype=dtype, device=device),
            nn.ReLU(),
            nn.MaxPool1d(pool_size, stride = pool_stride))       
        out1_size = int((1280 + 2*padding - (kernel_size - 1) - 1)/stride + 1)
        maxpool1_size = int((out1_size + 2*padding - (pool_size - 1) - 1)/pool_stride + 1)
        print ('maxpool1_size', maxpool1_size*2)
        
        self.layer2 = nn.Sequential(
            nn.Conv1d(1, 2, kernel_size, stride = stride, padding = padding, dtype=dtype, device=device),
            nn.BatchNorm1d(2, dtype=dtype, device=device),
            nn.ReLU(),
            nn.MaxPool1d(pool_size, stride = pool_stride))       
        out2_size = int((1280 + 2*padding - (kernel_size - 1) - 1)/stride + 1)
        maxpool2_size = int((out2_size + 2*padding - (pool_size - 1) - 1)/pool_stride + 1)
        print ('maxpool2_size', maxpool2_size*2)
        
        self.layer3 = nn.Sequential(
            nn.Conv1d(1, 2, kernel_size, stride = stride, padding = padding, dtype=dtype, device=device),
            nn.BatchNorm1d(2, dtype=dtype, device=device),
            nn.ReLU(),
            nn.MaxPool1d(pool_size, stride = pool_stride))       
        out3_size = int((1280 + 2*padding - (kernel_size - 1) - 1)/stride + 1)
        maxpool3_size = int((out3_size + 2*padding - (pool_size - 1) - 1)/pool_stride + 1)
        print ('maxpool3_size', maxpool3_size*2)
        
        self.layer4 = nn.Sequential(
            nn.Conv1d(1, 2, kernel_size, stride = stride, padding = padding, dtype=dtype, device=device),
            nn.BatchNorm1d(2, dtype=dtype, device=device),
            nn.ReLU(),
            nn.MaxPool1d(pool_size, stride = pool_stride))       
        out4_size = int((1280 + 2*padding - (kernel_size - 1) - 1)/stride + 1)
        maxpool4_size = int((out4_size + 2*padding - (pool_size - 1) - 1)/pool_stride + 1)
        print ('maxpool4_size', maxpool4_size*2)
        
        self.flatten = nn.Flatten()
        
        #self.drop1 = nn.Dropout(p=0.25)
        #self.fc1 = nn.Linear(maxpool1_size*2 + maxpool2_size*4, hidden_dim, dtype=dtype, device=device)
        #self.drop2 = nn.Dropout(p=0.25)
        #self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(maxpool1_size*2 + maxpool2_size*2 + maxpool3_size*2 + maxpool4_size*2, out_dim, dtype=dtype, device=device)        
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, input):

        input = input.reshape(-1,1,1280)
        
        out1 = self.layer1(input)  
        out1 = self.flatten(out1)
        
        out2 = self.layer2(input)
        out2 = self.flatten(out2)
        
        out3 = self.layer3(input)
        out3 = self.flatten(out3)
        
        out4 = self.layer4(input)
        out4 = self.flatten(out4)
        
        out = torch.cat([out1, out2, out3, out4], 1)
        out = self.dropout(out)
        
        #out = self.drop1(out)
        #out = self.fc1(out)
        #out = self.drop2(out)
        #out = self.relu1(out)
        out = self.fc2(out)
        
        return out
'''
class CNN4(nn.Module):

    def __init__(self, hidden_dim, out_dim, device, dtype, drop_out=0.15, kernel_size=3, stride=1, pool_size=3, pool_stride= 3, padding=0):
        super(CNN4, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 2, 1, stride = stride, padding = padding, dtype=dtype, device=device),
            nn.BatchNorm1d(2, dtype=dtype, device=device),
            nn.ReLU(),
            nn.MaxPool1d(pool_size, stride = pool_stride))       
        out1_size = int((1280 + 2*padding - (1 - 1) - 1)/stride + 1)
        maxpool1_size = int((out1_size + 2*padding - (pool_size - 1) - 1)/pool_stride + 1)
        print ('maxpool1_size', maxpool1_size*2)
        
        self.layer2 = nn.Sequential(
            nn.Conv1d(1, 2, kernel_size, stride = stride, padding = padding, dtype=dtype, device=device),
            nn.BatchNorm1d(2, dtype=dtype, device=device),
            nn.ReLU(),
            nn.MaxPool1d(pool_size, stride = pool_stride))       
        out2_size = int((1280 + 2*padding - (kernel_size - 1) - 1)/stride + 1)
        maxpool2_size = int((out2_size + 2*padding - (pool_size - 1) - 1)/pool_stride + 1)
        print ('maxpool2_size', maxpool2_size*2)
        
        self.layer3 = nn.Sequential(
            nn.Conv1d(1, 3, kernel_size, stride = stride, padding = padding, dtype=dtype, device=device),
            nn.BatchNorm1d(3, dtype=dtype, device=device),
            nn.ReLU(),
            nn.MaxPool1d(pool_size, stride = pool_stride))       
        out3_size = int((1280 + 2*padding - (kernel_size - 1) - 1)/stride + 1)
        maxpool3_size = int((out3_size + 2*padding - (pool_size - 1) - 1)/pool_stride + 1)
        print ('maxpool3_size', maxpool3_size*3)
        
        self.layer4 = nn.Sequential(
            nn.Conv1d(1, 2, 5, stride = stride, padding = padding, dtype=dtype, device=device),
            nn.BatchNorm1d(2, dtype=dtype, device=device),
            nn.ReLU(),
            nn.MaxPool1d(pool_size, stride = pool_stride))       
        out4_size = int((1280 + 2*padding - (5 - 1) - 1)/stride + 1)
        maxpool4_size = int((out4_size + 2*padding - (pool_size - 1) - 1)/pool_stride + 1)
        print ('maxpool4_size', maxpool4_size*2)
        
        self.flatten = nn.Flatten()
        
        #self.drop1 = nn.Dropout(p=0.25)
        #self.fc1 = nn.Linear(maxpool1_size*2 + maxpool2_size*4, hidden_dim, dtype=dtype, device=device)
        #self.drop2 = nn.Dropout(p=0.25)
        #self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(maxpool1_size*2 + maxpool2_size*2 + maxpool3_size*3 + maxpool4_size*2, out_dim, dtype=dtype, device=device)        
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, input):

        input = input.reshape(-1,1,1280)
        
        out1 = self.layer1(input)  
        out1 = self.flatten(out1)
        
        out2 = self.layer2(input)
        out2 = self.flatten(out2)
        
        out3 = self.layer3(input)
        out3 = self.flatten(out3)
        
        out4 = self.layer4(input)
        out4 = self.flatten(out4)
        
        out = torch.cat([out1, out2, out3, out4], 1)
        out = self.dropout(out)
        
        #out = self.drop1(out)
        #out = self.fc1(out)
        #out = self.drop2(out)
        #out = self.relu1(out)
        out = self.fc2(out)
        
        return out        
        
class CNN5(nn.Module):

    def __init__(self, hidden_dim, out_dim, device, dtype, drop_out=0.15, kernel_size=3, stride=1, pool_size=3, pool_stride= 3, padding=0):
        super(CNN5, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 1, 1, stride = stride, padding = padding, dtype=dtype, device=device),
            nn.BatchNorm1d(1, dtype=dtype, device=device),
            nn.ReLU(),
            nn.MaxPool1d(pool_size, stride = pool_stride))       
        out1_size = int((1280 + 2*padding - (1 - 1) - 1)/stride + 1)
        maxpool1_size = int((out1_size + 2*padding - (pool_size - 1) - 1)/pool_stride + 1)
        print ('maxpool1_size', maxpool1_size*1)
        
        self.layer2 = nn.Sequential(
            nn.Conv1d(1, 2, 1, stride = stride, padding = padding, dtype=dtype, device=device),
            nn.BatchNorm1d(2, dtype=dtype, device=device),
            nn.ReLU(),
            nn.MaxPool1d(pool_size, stride = pool_stride))       
        out2_size = int((1280 + 2*padding - (1 - 1) - 1)/stride + 1)
        maxpool2_size = int((out2_size + 2*padding - (pool_size - 1) - 1)/pool_stride + 1)
        print ('maxpool2_size', maxpool2_size*2)
        
        self.layer3 = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size, stride = stride, padding = padding, dtype=dtype, device=device),
            nn.BatchNorm1d(1, dtype=dtype, device=device),
            nn.ReLU(),
            nn.MaxPool1d(pool_size, stride = pool_stride))       
        out3_size = int((1280 + 2*padding - (kernel_size - 1) - 1)/stride + 1)
        maxpool3_size = int((out3_size + 2*padding - (pool_size - 1) - 1)/pool_stride + 1)
        print ('maxpool3_size', maxpool3_size*1)
        
        self.layer4 = nn.Sequential(
            nn.Conv1d(1, 2, kernel_size, stride = stride, padding = padding, dtype=dtype, device=device),
            nn.BatchNorm1d(2, dtype=dtype, device=device),
            nn.ReLU(),
            nn.MaxPool1d(pool_size, stride = pool_stride))       
        out4_size = int((1280 + 2*padding - (kernel_size - 1) - 1)/stride + 1)
        maxpool4_size = int((out4_size + 2*padding - (pool_size - 1) - 1)/pool_stride + 1)
        print ('maxpool4_size', maxpool4_size*2)
        
        self.layer5 = nn.Sequential(
            nn.Conv1d(1, 2, 5, stride = stride, padding = padding, dtype=dtype, device=device),
            nn.BatchNorm1d(2, dtype=dtype, device=device),
            nn.ReLU(),
            nn.MaxPool1d(pool_size, stride = pool_stride))       
        out5_size = int((1280 + 2*padding - (5 - 1) - 1)/stride + 1)
        maxpool5_size = int((out5_size + 2*padding - (pool_size - 1) - 1)/pool_stride + 1)
        print ('maxpool5_size', maxpool5_size*2)
        
        self.flatten = nn.Flatten()
        
        #self.drop1 = nn.Dropout(p=0.25)
        #self.fc1 = nn.Linear(maxpool1_size*2 + maxpool2_size*4, hidden_dim, dtype=dtype, device=device)
        #self.drop2 = nn.Dropout(p=0.25)
        #self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(maxpool1_size*1 + maxpool2_size*2 + maxpool3_size*1 + maxpool4_size*2 + maxpool5_size*2, out_dim, dtype=dtype, device=device)        
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, input):

        input = input.reshape(-1,1,1280)
        
        out1 = self.layer1(input)  
        out1 = self.flatten(out1)
        
        out2 = self.layer2(input)
        out2 = self.flatten(out2)
        
        out3 = self.layer3(input)
        out3 = self.flatten(out3)
        
        out4 = self.layer4(input)
        out4 = self.flatten(out4)
        
        out5 = self.layer5(input)
        out5 = self.flatten(out5)
        
        out = torch.cat([out1, out2, out3, out4, out5], 1)
        out = self.dropout(out)
        
        #out = self.drop1(out)
        #out = self.fc1(out)
        #out = self.drop2(out)
        #out = self.relu1(out)
        out = self.fc2(out)
        
        return out   
        
class CNN_SA(nn.Module):

    def __init__(self, hidden_size, out_dim, device, dtype, num_attention_heads=4, drop_out=0.1, kernel_size=3, stride=1, pool_size=3, pool_stride= 3, padding=0):
        super(CNN_SA, self).__init__()
        
        #hidden_size = 128
        if hidden_size % num_attention_heads != 0 :
            raise ValueError(
                "the hidden size %d is not a multiple of the number of attention heads"
                "%d" % (hidden_size, num_attention_heads)
            )

        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 2, kernel_size, stride = stride, padding = padding, dtype=dtype, device=device),
            nn.BatchNorm1d(2, dtype=dtype, device=device),
            nn.ReLU(),
            nn.MaxPool1d(pool_size, stride = pool_stride))       
        out1_size = int((1280 + 2*padding - (kernel_size - 1) - 1)/stride + 1)
        maxpool1_size = int((out1_size + 2*padding - (pool_size - 1) - 1)/pool_stride + 1)
        print ('maxpool1_size', maxpool1_size*2)#426*2
        
        self.layer2 = nn.Sequential(
            nn.Conv1d(1, 2, kernel_size, stride = stride, padding = padding, dtype=dtype, device=device),
            nn.BatchNorm1d(2, dtype=dtype, device=device),
            nn.ReLU(),
            nn.MaxPool1d(pool_size, stride = pool_stride))       
        out2_size = int((1280 + 2*padding - (kernel_size - 1) - 1)/stride + 1)
        maxpool2_size = int((out2_size + 2*padding - (pool_size - 1) - 1)/pool_stride + 1)
        print ('maxpool2_size', maxpool2_size*2)
        
        self.layer3 = nn.Sequential(
            nn.Conv1d(1, 2, kernel_size, stride = stride, padding = padding, dtype=dtype, device=device),
            nn.BatchNorm1d(2, dtype=dtype, device=device),
            nn.ReLU(),
            nn.MaxPool1d(pool_size, stride = pool_stride))       
        out3_size = int((1280 + 2*padding - (kernel_size - 1) - 1)/stride + 1)
        maxpool3_size = int((out3_size + 2*padding - (pool_size - 1) - 1)/pool_stride + 1)
        print ('maxpool3_size', maxpool3_size*2)
        
        self.flatten = nn.Flatten()#426*6=2556

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.key_layer = nn.Linear(maxpool1_size*2 + maxpool2_size*2 + maxpool3_size*2, hidden_size, dtype=dtype, device=device)
        self.query_layer = nn.Linear(maxpool1_size*2 + maxpool2_size*2 + maxpool3_size*2, hidden_size, dtype=dtype, device=device)
        self.value_layer = nn.Linear(maxpool1_size*2 + maxpool2_size*2 + maxpool3_size*2, hidden_size, dtype=dtype, device=device)
        
        #self.fc2 = nn.Linear(maxpool1_size*2 + maxpool2_size*2 + maxpool3_size*2, out_dim, dtype=dtype, device=device)        
        self.dropout = nn.Dropout(p=drop_out)

    def trans_to_multiple_heads(self, x):
        new_size = x.size()[ : -1] + (self.num_attention_heads, self.attention_head_size)
        print(new_size)
        x = x.view(new_size)
        print(x.shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        x = x.reshape(-1,1,1280)
        
        out1 = self.layer1(x)  
        out1 = self.flatten(out1)
        
        out2 = self.layer2(x)
        out2 = self.flatten(out2)
        
        out3 = self.layer3(x)
        out3 = self.flatten(out3)
        
        out = torch.cat([out1, out2, out3], 1)
        x = self.dropout(out)
        print(x.size()[1])
        x = x.reshape(-1,1,x.size()[1])
        
        key = self.key_layer(x)
        query = self.query_layer(x)
        value = self.value_layer(x)
        print(key.shape)

        key_heads = self.trans_to_multiple_heads(key)
        query_heads = self.trans_to_multiple_heads(query)
        value_heads = self.trans_to_multiple_heads(value)

        attention_scores = torch.matmul(query_heads, key_heads.permute(0, 1, 3, 2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = F.softmax(attention_scores, dim = -1)

        context = torch.matmul(attention_probs, value_heads)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_size = context.size()[ : -2] + (self.all_head_size , )
        context = context.view(*new_size)
        
        context = torch.squeeze(context)
        print(context.shape)
        return context


class BatchNormNet(nn.Module):
    def __init__(self, hidden_dim, out_dim, device, dtype, drop_out=0.1):
        super(BatchNormNet, self).__init__()
        self.hidden_dim1 = hidden_dim
        self.out_dim = out_dim
        self.drop_out = drop_out
        self.device = device
        self.dtype = dtype

        self.fc1 = nn.Linear(1280, hidden_dim, dtype=dtype, device=device)
        self.bn1 = nn.BatchNorm1d(hidden_dim, dtype=dtype, device=device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim,
                             dtype=dtype, device=device)
        self.bn2 = nn.BatchNorm1d(hidden_dim, dtype=dtype, device=device)
        self.fc3 = nn.Linear(hidden_dim, out_dim, dtype=dtype, device=device)
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        x = self.dropout(self.bn1(self.fc1(x)))
        x = torch.relu(x)
        x = self.dropout(self.bn2(self.fc2(x)))
        x = torch.relu(x)
        x = self.fc3(x)
        return x
    
 
class InstanceNorm(nn.Module):
    def __init__(self, hidden_dim, out_dim, device, dtype, drop_out=0.1):
        super(InstanceNorm, self).__init__()
        self.hidden_dim1 = hidden_dim
        self.out_dim = out_dim
        self.drop_out = drop_out
        self.device = device
        self.dtype = dtype

        self.fc1 = nn.Linear(1280, hidden_dim, dtype=dtype, device=device)
        self.in1 = nn.InstanceNorm1d(hidden_dim, dtype=dtype, device=device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim,
                             dtype=dtype, device=device)
        self.in2 = nn.InstanceNorm1d(hidden_dim, dtype=dtype, device=device)
        self.fc3 = nn.Linear(hidden_dim, out_dim, dtype=dtype, device=device)
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        x = self.dropout(self.in1(self.fc1(x)))
        x = torch.relu(x)
        x = self.dropout(self.in2(self.fc2(x)))
        x = torch.relu(x)
        x = self.fc3(x)
        return x
