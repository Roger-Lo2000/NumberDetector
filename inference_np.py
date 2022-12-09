import cv2
import numpy as np
import time

def relu(x):
    return (np.maximum(0,x))
def inference(img,fc1_weight,fc1_bias,fc2_weight,fc2_bias):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
    img = cv2.resize(img,(28,28),interpolation=cv2.INTER_AREA)
    img = img.reshape(-1, 28*28)
    img = img.astype(np.float32)
    hidden1 = np.zeros(len(fc1_weight),dtype = np.float32)
    out = np.zeros(len(fc2_weight),dtype = np.float32)

    for i in range(len(fc1_weight)):
        hidden1[i] = np.sum(img[0] * fc1_weight[i])
    hidden1 += fc1_bias
    hidden = relu(hidden1)
    for i in range(len(fc2_weight)):
        out[i] = np.sum(hidden * fc2_weight[i])
    return np.argmax(out)

# label = ['0','1','2','3','4','5','6','7','8','9',       
#          'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
#          'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',]
#label = ['N/A','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
label = ['0','1','2','3','4','5','6','7','8','9'] 

if __name__ == '__main__':
    fc1_weight = np.load('weight/fc1_weight.npy')
    fc2_weight = np.load('weight/fc2_weight.npy')
    fc1_bias = np.load('weight/fc1_bias.npy')
    fc2_bias = np.load('weight/fc2_bias.npy')
    img = cv2.imread('44.jpg',)
    ctime = time.time()
    result = inference(img,fc1_weight,fc1_bias,fc2_weight,fc2_bias)
    t = time.time() - ctime
    print(label[result],t)