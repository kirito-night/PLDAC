from __future__ import print_function
import cv2 as cv
from track import *
import numpy as np
import pandas as pd
from scipy.signal import convolve2d
 
'''
该代码尝试使用背景差分法,完成了固定摄像头中,动态物体的提取。
'''
#定义输出文件
Fourcc = cv.VideoWriter_fourcc(*'MP4V')
out = cv.VideoWriter('result/nid/method1_location_dectection.mp4',Fourcc, 25, (1440,1080),False)
tracker = EuclideanDistTracker() 
#mat=[]
mat_1=[]

#有两种算法可选,KNN和MOG2,下面的代码使用KNN作为尝试
algo='KNN'
if algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()

# Useful functions for this work
def orientation(Ix, Iy, Ig):
    """ Array[n,m]**3 -> Array[n,m]
        Returns an image of orientation.
    """
    n, m = Ix.shape
    x = np.arange(4)*np.pi/4
    ori = np.stack((np.cos(x), np.sin(x)), axis=1)
    O = np.zeros(Ix.shape)
    for i in range(n):
        for j in range(m):
            if Ig[i, j] > 0:
                v = np.array([Ix[i, j], -Iy[i, j]])/Ig[i, j]
                if Iy[i, j] > 0: v = -v
                prod = np.matmul(ori, v)
                maxi = prod.max()
                imax = np.nonzero(prod == maxi)
                O[i, j] = imax[0][0]+1
    return O

def gaussianKernel(sigma):
    """ double -> Array
        return a gaussian kernel of standard deviation sigma
    """
    n2 = int(np.ceil(3*sigma))
    x,y = np.meshgrid(np.arange(-n2,n2+1),np.arange(-n2,n2+1))
    kern =  np.exp(-(x**2+y**2)/(2*sigma*sigma))
    return kern/kern.sum()

def computeR(image,scale,kappa):
    """ Array[n, m]*float*float->Array[n, m]
    """
    # compute the derivatives
    Ix = convolve2d(image, np.array([[-1, 0, 1]]), mode='same')
    Iy = convolve2d(image, np.array([[-1, 0, 1]]).T, mode='same')
    Ixx = Ix**2
    Iyy = Iy**2
    Ixy = Ix*Iy
    # compute the gaussian kernel
    kern = gaussianKernel(scale)
    # compute the sum of the squares of the derivatives
    Sxx = convolve2d(Ixx, kern, mode='same')
    Syy = convolve2d(Iyy, kern, mode='same')
    Sxy = convolve2d(Ixy, kern, mode='same')
    # compute the R matrix
    R = (Sxx*Syy - Sxy**2) - kappa*(Sxx+Syy)**2
    return R

def rnms(image_harris,Rbin):
    """ Array[n, m] -> Array[n, m] 
    """
    # compute the size of the image
    n,m = image_harris.shape
    # create a new image
    image_rnms = np.zeros((n,m))
    # for each pixel
    for i in range(1,n-1):
        for j in range(1,m-1):
            # if the pixel is a local maximum,check if the pixel is the local maximum of its 8-neighborhood
            if Rbin[i,j] == 1 and np.max(image_harris[i-1:i+2, j-1:j+2]) == image_harris[i,j]:
                image_rnms[i,j] = 1
    return image_rnms

def cornerDetector(image, sigma, kappa, thres):
    """ Array[n, m]*float*float*float -> Array[n, m]
    """
    R = computeR(image, sigma, kappa)
    max,min=R.max(),R.min()
    Rnorm=(R-min)/(max-min)
    Rbin = thresholdR(Rnorm, thres)
    image_rnms = rnms(Rnorm, Rbin)
    return image_rnms

def thresholdR(R, thres):
    """ Array[n, m] * float -> Array[n, m]
    """
    return(np.where(R>thres, 1, 0))


#打开一个视频文件
capture = cv.VideoCapture(cv.samples.findFileOrKeep('resource/video_juste_nid-test.ts'))
#判断视频是否读取成功
if not capture.isOpened():
    print('Unable to open')
    exit(0)


# variables
sigma=5
kappa=0.04
thres=0.3
time=-1

#逐帧读取视频,进行相关分析
while (True):
    time+=1
    #读取视频的第一帧
    ret, frame = capture.read()
    if frame is None:
        break
    #使用定义的backSub对象,输入新的一帧frame,生成背景蒙版
    #frame=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    frame=np.where(frame[:,:,2]>100,255,0).astype(np.uint8)
    #frame=cv.GaussianBlur(frame,(5,5),2)


    image_rnms = cornerDetector(frame, sigma, kappa, thres)
    n2=int(np.ceil(3*sigma))
    x,y=np.meshgrid(np.arange(-n2,n2+1),np.arange(-n2,n2+1))
    corners=np.argwhere(image_rnms==1)

    body=[]
    for pos1 in corners:
        for pos2 in corners:
            mean=np.mean([pos1,pos2],axis=0)
            if np.linalg.norm(pos1-pos2)<50 and ((pos1-pos2) != [0,0]).all:
                body.append([pos1[0],pos1[1],pos2[0],pos2[1]])


    # 物体追踪
    boxer_ids = tracker.update(body)  # 同一个物体会有相同的ID
    # print(boxer_ids)
    for box_id in boxer_ids:
        x, y, w, h, id = box_id
        cv.putText(frame, "Obj" + str(id), (x, y - 15), cv.FONT_ITALIC, 0.7, (255, 255, 255), 2)
        cv.rectangle(frame, (x, y), (w, h), (255, 255, 255), 2)  # 根据移动物体的轮廓添加boundingbox
        mat_1.append({
            'time': time,
            'id': id,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'center_x': int((x + w)/2),
            'center_y': int((y + h)/2)
        })
            
    #展示视频中的物体,三个窗口分别表示原视频.背景.移动目标
    #if len(detections)!=0:
    #    detections=np.array(detections)
        #print(detections[0,:2]+0.5*detections[0,2:])
                # Ajouter la fourmi à la liste

    cv.imshow('Frame', frame)
    #cv.imshow('FG Mask', fgMask)
    #cv.imshow('Object',Object)

    #将当前帧写入输出文件
    frame.dtype=np.uint8
    out.write(frame)
    #mat.append(boxer_ids)
    #mat_1.append(detections[:,:2]+0.5*detections[:,2:])

    #每帧展示结束,等待30毫秒
    keyboard = cv.waitKey(25)
    #按q推出程序
    if  keyboard == 27:
        break


#存入npy文件
#mat=np.array(mat,dtype=object)
#np.save('result/boite/location_dectection.npy',mat)
DataFrame = pd.DataFrame(mat_1)
DataFrame.to_csv('result/nid/method1_location_dectection.csv', index=False, sep=',')

#释放资源
capture.release()
out.release()
cv.destroyAllWindows()


