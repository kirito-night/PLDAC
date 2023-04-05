from __future__ import print_function
import cv2 as cv
import argparse
from track import *
import numpy as np
import pandas as pd
 
'''
该代码尝试使用背景差分法,完成了固定摄像头中,动态物体的提取。
'''
#定义输出文件
Fourcc = cv.VideoWriter_fourcc(*'MPEG')
out = cv.VideoWriter('result/boite/location_dectection.ts',Fourcc, 25, (1440,1080),False)
tracker = EuclideanDistTracker() 
#mat=[]
mat_1=[]

#有两种算法可选,KNN和MOG2,下面的代码使用KNN作为尝试
algo='KNN'
if algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()


#打开一个视频文件
capture = cv.VideoCapture(cv.samples.findFileOrKeep('resource/video_boite_entiere-test.ts'))
#判断视频是否读取成功
if not capture.isOpened():
    print('Unable to open')
    exit(0)




time=-1

#逐帧读取视频,进行相关分析
while (True):
    time+=1
    #读取视频的第一帧
    ret, frame = capture.read()
    if frame is None:
        break
    #使用定义的backSub对象,输入新的一帧frame,生成背景蒙版
    frame=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    #frame=cv.GaussianBlur(frame,(5,5),2)
    #frame=np.where(frame[:,:,2]>110,255,0)

    fgMask = backSub.apply(frame)
    
    #将原视频的当前帧和蒙版做相加运算,将前景物体提取出来
    Object=cv.add(frame,frame,mask=fgMask)
    Object=cv.blur(Object,(5,5))
    _ ,Object = cv.threshold(Object, 80, 255, cv.THRESH_BINARY)
    Object=cv.blur(Object,(5,5))
    #_ ,Object = cv.threshold(Object, 80, 255, cv.THRESH_BINARY)
    #Object=cv.erode(Object,None,iterations=1)
    #Object=cv.dilate(Object,None,iterations=1)
    
   

    contours, _ = cv.findContours(Object, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # 找到视频中物体的轮廓



    detections = []  # 用于存放boundingbox的起始点坐标.宽.高
    for cnt in contours:
        # 计算出每个轮廓内部的面积,并根据面积的大小去除那些不必要的噪声（比如树.草等等）
        area = cv.contourArea(cnt)
        if area > 100 and area < 300:
            #cv.drawContours(Object, [cnt], -1, (0, 255, 0), 2)  # 画出移动物体的轮廓
            x, y, w, h = cv.boundingRect(cnt)
            detections.append([x, y, w, h])
            #cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)  # 画出移动物体的外接矩形
    #print(len(detections))



    # 物体追踪
    boxer_ids = tracker.update(detections)  # 同一个物体会有相同的ID
    # print(boxer_ids)
    for box_id in boxer_ids:
        x, y, w, h, id = box_id
        cv.putText(frame, "Obj" + str(id), (x, y - 15), cv.FONT_ITALIC, 0.7, (255, 255, 255), 2)
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)  # 根据移动物体的轮廓添加boundingbox
        mat_1.append({
            'time': time,
            'id': id,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'center_x': int(x + w/2),
            'center_y': int(y + h/2)
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
DataFrame.to_csv('result/boite/location_dectection.csv', index=False, sep=',')

#释放资源
capture.release()
out.release()
cv.destroyAllWindows()


