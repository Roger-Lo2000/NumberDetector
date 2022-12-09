import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2
import time
import numpy as np
# Device configuration
device = torch.device('cpu')
print('using gpu to do inference')
#print(torch.cuda.is_available())
hidden_size1 = 500
input_size = 784 # 28x28
num_classes = 10
class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(784, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size1)
        self.fc3 = nn.Linear(hidden_size1, hidden_size1)
        self.fc4 = nn.Linear(hidden_size1, hidden_size1)
        self.fc5 = nn.Linear(hidden_size1, num_classes)  

    
    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        out = self.fc5(out)
        return out

model = NeuralNet(input_size, num_classes).to(device)
print(model)
# model = NeuralNet(input_size, hidden_size, num_classes).to(device)
#label = ['N/A','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
label = ['0','1','2','3','4','5','6','7','8','9'] 
print(len(label))
img = cv2.imread('9.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_, img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
img = cv2.resize(img,(28,28),interpolation=cv2.INTER_AREA)
cv2.imshow('',img)
cv2.waitKey(0)
transform = transforms.Compose([transforms.ToTensor()])
img = img.reshape(-1, 28*28)
img = transform(img).to(device)
model.load_state_dict(torch.load('model_fc.pth'))
print(model)
#model.to(device)
model.eval()
ctime = time.time()
ans = model(img).squeeze()
_,predict = torch.max(ans,0)
#print(ans)
ntime = time.time()
print(predict)
print('predict number: ' + label[predict.item()])
print('spend',ntime - ctime ,'to do the inference')
