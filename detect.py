import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

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
    ## print(width,height)
    if(400 > width > 200 and 460 > height > 300):
        return True
    else:
        return False

def project_row(img):
    rows = img.shape[0]
    cols = img.shape[1]
    l = np.zeros(rows,dtype = np.int32)
    x = np.arange(0,rows,1,dtype = np.int32)
    for i in range(rows):
        for j in range(cols):
            if(img[i][j] == 0):
                l[i] += 1
    # plt.plot(x,l)
    # plt.show()
    for i in range(rows):
        if(l[i] > 1):
            l[i] = 1
        else:
            l[i] = 0
    # plt.plot(x,l)
    # plt.show()

    return l
def project_col(img):
    rows = img.shape[0]
    cols = img.shape[1]
    l = np.zeros(cols,dtype = np.int32)
    x = np.arange(0,cols,1,dtype = np.int32)
    for i in range(cols):
        for j in range(rows):
            if(img[j][i] == 0):
                l[i] += 1
    for i in range(cols):
        if(l[i] > 1):
            l[i] = 1
        else:
            l[i] = 0
    # plt.plot(x,l)
    # plt.show()
    return l

def split_row(row_arr):
    val = 0
    bound = []
    for i in range(len(row_arr)):
        if(val != row_arr[i]):
            bound.append(i)
            val = row_arr[i]
    #print(bound)
    return bound

def split_col(col_arr):
    val = 0
    bound = []
    for i in range(len(col_arr)):
        if(val != col_arr[i]):
            bound.append(i)
            val = col_arr[i]
    ## print(bound)
    return bound
def split_img_row(img,bound):
    num = len(bound) // 2
    s = []
    for i in range(0,num-2,2):
        s.append(img[bound[i]-2:bound[i+1]+2,:])
    
def split_img(img,bound):
    # num = len(bound) // 2  
    # print(num)    
    s = []
    #print(bound)
    for i in range(0,len(bound),2):
        s.append(img[:,bound[i] - 2:bound[i+1] + 2])
    for i in range(len(s)):
        if(s[i].size != 0):
            cv2.imshow('sub-img',s[i])
            cv2.imwrite(str(i)+'.jpg',s[i])
    # exit()
if __name__ == '__main__':
    
    cap = cv2.VideoCapture(0)
    t = np.zeros((512,512),dtype=np.uint8)  
    th= 50
    cnt = 0
    while(True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
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
                    cnt += 1
                    if(cnt > th):
                        M = cv2.getPerspectiveTransform(p, dst)
                        t = cv2.warpPerspective(frame,M,(int(dst[3][0]),int(dst[3][1])))
                    break
        cv2.imshow('frame',frame)

        # print(cnt)
        # print(t.all())
        if(cnt > th and t.all() and t.size != 0):
            g = cv2.cvtColor(t,cv2.COLOR_BGR2GRAY)
            blur = cv2.gaussianblur(g,(9,9),0)
            binary = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,13,2)
            dilate = cv2.dilate(binary,kernel_d,iterations = 1)
            erode_t = cv2.erode(dilate,kernel_e,iterations = 3)
            erode_t = erode_t[20:erode_t.shape[0] - 20,:]
            row_arr = project_row(erode_t)
            col_arr = project_col(erode_t)
            bound_row = split_row(row_arr)
            bound_col = split_col(col_arr)
            # cv2.imshow('transform',erode_t)
            if(len(bound_row) == 2):
                t = t[bound_row[0]+20:bound_row[1]+20]
                img_set = split_img(t,bound_col)  
            if(t.size != 0):
               cv2.imshow('transform',t)
            #cv2.waitKey(0)
        
        if(cnt > 200):
            ## t = np.zeros((512,512),dtype = np.uint8)
            ## cv2.destroyWindow('transform')
            cnt = 0

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break
    cap.release()
    cv2.destroyAllWindows()
