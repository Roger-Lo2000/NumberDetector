import cv2
import numpy as np
import time

def relu(x):
    return (np.maximum(0,x))
def inference(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
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

fc1_weight = np.load('weight/fc1_weight.npy')
fc2_weight = np.load('weight/fc2_weight.npy')
fc1_bias = np.load('weight/fc1_bias.npy')
fc2_bias = np.load('weight/fc2_bias.npy')
img = cv2.imread('6.jpg')

ctime = time.time()
result = inference(img)
t = time.time() - ctime

print(result,t)


