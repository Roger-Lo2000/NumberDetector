import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2
import time
# Device configuration
device = torch.device('cpu')
print('using gpu to do inference')
#print(torch.cuda.is_available())
hidden_size1 = 500
hidden_size2 = 500
input_size = 784 # 28x28
num_classes = 10
class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.conv1 = nn.Conv2d(1,6,3)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,3)

        self.fc1 = nn.Linear(11*11*16, hidden_size1) 
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)  
        self.fc3 = nn.Linear(hidden_size2, num_classes)  

    
    def forward(self, x):
        out = self.pool(F.relu(self.conv1(x)))
        out = F.relu(self.conv2(out))
        out = out.view(-1,11*11*16)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        # no activation and no softmax at the end
        return out

model = NeuralNet(input_size, num_classes).to(device)
print(model)
# model = NeuralNet(input_size, hidden_size, num_classes).to(device)
lable = ['0','1','2','3','4','5','6','7','8','9']
img = cv2.imread('9.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
transform = transforms.Compose([transforms.ToTensor()])
#img = img.reshape(-1, 28*28)
img = transform(img).to(device)
model.load_state_dict(torch.load('model_CNN.pth'))
#model.to(device)
model.eval()
ctime = time.time()
ans = model(img).squeeze()
_,predict = torch.max(ans,0)
ntime = time.time()
for param in model.parameters():
    print(param)
print('predict number: ' + lable[predict.item()])
print('spend',ntime - ctime ,'to do the inference')
