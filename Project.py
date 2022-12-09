import cv2
import numpy as np
import matplotlib.pyplot as plt
import inference_np as inp
import time
def sort(app):
    app = np.squeeze(app)
    rest = np.copy(app)
    new_app = np.empty((4,2))
    dst = np.empty((4,2))
    min_index = 0
    max_index = 0
    mini = app[min_index][0]**2 + app[min_index][1]**2
    maxi = app[max_index][0]**2 + app[max_index][1]**2
    for i in range(1,4):
        if(mini > app[i][0]**2 + app[i][1]**2 ):
            mini = app[i][0]**2 + app[i][1]**2 
            min_index = i 
        if(maxi < app[i][0]**2 + app[i][1]**2):
            maxi = app[i][0]**2 + app[i][1]**2
            max_index = i
    rest = np.delete(rest,(min_index,max_index),axis=0)
    if(rest[0][0] > rest[1][0]):
        tr = rest[0]
        bl = rest[1]
    else:
        tr = rest[1]
        bl = rest[0]
    new_app[0] = app[min_index]
    new_app[1] = tr
    new_app[2] = bl
    new_app[3] = app[max_index] 
    width = (new_app[1][0] - new_app[0][0] + (new_app[3][0] - new_app[2][0]))//2
    height = (new_app[2][1] - new_app[0][1] + new_app[3][1] - new_app[1][1])//2
    dst[0] = [0,0]
    dst[1] = [width,0]
    dst[2] = [0,height]
    dst[3] = [width,height]
    new_app = new_app.astype(np.float32)
    dst = dst.astype(np.float32)

    return new_app, dst

def calc_w_h(src):
    width = (src[1][0] - src[0][0] + (src[3][0] - src[2][0]))//2
    height = (src[2][1] - src[0][1] + src[3][1] - src[1][1])//2
    #print(width,height)
    if(height == 0):
        return False
    if(1.6 > width / height > 1.2):
        return True
    else:
        return False

fc1_weight = np.load('weight/fc1_weight.npy')
fc2_weight = np.load('weight/fc2_weight.npy')
fc1_bias = np.load('weight/fc1_bias.npy')
fc2_bias = np.load('weight/fc2_bias.npy')
# label = ['0','1','2','3','4','5','6','7','8','9',       
#          'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
#          'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',]
label = ['0','1','2','3','4','5','6','7','8','9'] 
#label = ['N/A','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']         
if __name__ == '__main__':
    
    cap = cv2.VideoCapture(0)
    while(True):
        ctime = time.time()
        ret, frame = cap.read()
        img_t = np.zeros(1)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,13,2)
        edge = cv2.Canny(gray,10,100)
        kernel_d = np.ones((5,5),np.uint8)
        kernel_e = np.ones((3,3),np.uint8)
        dilate = cv2.dilate(edge,kernel_d,iterations = 1)
        erode = cv2.erode(dilate,kernel_e,iterations = 1)
        contours,hierarchy = cv2.findContours(erode,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        #cv2.drawContours(frame,contours,-1,(255,255,0),3,lineType = cv2.LINE_AA)
        for obj in contours:
            area = cv2.contourArea(obj)
            cv2.drawContours(erode,obj,-1,(255,0,0),4)
            perimeter = cv2.arcLength(obj,True)
            approx = cv2.approxPolyDP(obj,0.02*perimeter,True)
            if(approx.all() and len(approx) == 4):     
                p,dst = sort(approx)    
                if(calc_w_h(p)):
                    M = cv2.getPerspectiveTransform(p, dst)
                    img_t = cv2.warpPerspective(frame,M,(int(dst[3][0]),int(dst[3][1])))
                    gray = cv2.cvtColor(img_t,cv2.COLOR_BGR2GRAY)
                    _, binary = cv2.threshold(gray,80,255,cv2.THRESH_BINARY_INV)
                    binary = cv2.resize(binary,(1920,1080),interpolation=cv2.INTER_CUBIC)
                    img_t = cv2.resize(img_t,(1920,1080),interpolation=cv2.INTER_CUBIC)       
        # w=0
        if(not np.all(img_t == 0)):
            contours_list,hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 
            for cnt in contours_list:
                x,y,w,h = cv2.boundingRect(cnt)
                if(500 > w > 28 and 500 > h > 28):                 
                    sub_img = img_t[y:y+h,x:x+w,:]
                    result = inp.inference(sub_img,fc1_weight,fc1_bias,fc2_weight,fc2_bias)
                    cv2.putText(img_t, label[result], (x, y), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.rectangle(img_t,(x,y),(x+w,y+h),(0, 255, 255),2)
            cv2.putText(img_t, 'fps: '+str(round(1/(time.time() - ctime),3)), (0,30), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow('img_t', img_t)

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break
    cap.release()
    cv2.destroyAllWindows()