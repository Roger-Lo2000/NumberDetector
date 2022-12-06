import cv2
import numpy as np
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
    #print(l)
    # plt.plot(x,l)
    # plt.show()
    for i in range(rows):
        if(l[i] > 5):
            l[i] = 1
        else:
            l[i] = 0


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
    # plt.plot(x,l)
    # plt.show()
    for i in range(cols):
        if(l[i] > 2):
            l[i] = 1
        else:
            l[i] = 0
    return l

def split_row(img,row_arr):
    val = 0
    l = [] 
    index = 0
    bound = [0,0]
    img_set = []
    for i in range(len(row_arr)):
        if(val != row_arr[i]):
            if(index == 0):
                val = row_arr[i]
                bound[index] = i
                index = 1
            elif(index == 1):
                val = row_arr[i]
                bound[index] = i
                index = 0
                l.append(bound.copy())
    if(index == 1):
        bound[index] = len(row_arr)
        l.append(bound.copy())

    for i in range(len(l)):
        img_set.append(img[l[i][0]:l[i][1],:])
    return img_set

def split_col(img,col_arr):
    val = 0
    l = [] 
    index = 0
    bound = [0,0]
    img_set = []
    for i in range(len(col_arr)):
        if(val != col_arr[i]):
            if(index == 0):
                val = col_arr[i]
                bound[index] = i
                index = 1
            elif(index == 1):
                val = col_arr[i]
                bound[index] = i
                index = 0
                l.append(bound.copy())
    if(index == 1):
        bound[index] = len(col_arr)
        l.append(bound.copy())
    for i in range(len(l)):
        img_set.append(img[:,l[i][0]:l[i][1]])
    return img_set
if __name__ == '__main__':
    
    cap = cv2.VideoCapture(0)
    while(True):
        ret, frame = cap.read()
        cv2.imshow('',frame)

        if(cv2.waitKey(1) & 0xFF == ord('y')):
            img = np.copy(frame)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
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
                        img = cv2.warpPerspective(gray,M,(int(dst[3][0]),int(dst[3][1])))
                        #img = cv2.dilate(img,kernel_d,iterations = 1)
                        #img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,13,2)
                        _, img = cv2.threshold(img,120,255,cv2.THRESH_BINARY)
                        img = img[10:img.shape[0] - 10,:]
                        img = img[:,10:img.shape[1]-10]
            row_arr = project_row(img)
            img_set_row = split_row(img,row_arr)
            img_set = []
            for i in range(len(img_set_row)):
                col_arr = project_col(img_set_row[i])
                img_set_col = split_col(img_set_row[i],col_arr)
                img_set.append(img_set_col)
            
            for i in range(len(img_set)):
                for j in range(len(img_set[i])):
                    cv2.imshow(str(i)+'_'+str(j),img_set[i][j])                   
            cv2.imshow('select frame',img)
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break
    cap.release()
    cv2.destroyAllWindows()