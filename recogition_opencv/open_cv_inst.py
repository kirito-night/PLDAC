from __future__ import print_function
import cv2 as cv
import argparse
from track import *
import numpy as np
import pandas as pd

'''
该代码尝试使用背景差分法,完成了固定摄像头中,动态物体的提取。
'''
# 定义输出文件
Fourcc = cv.VideoWriter_fourcc(*'MPEG')
out = cv.VideoWriter(
    'result/boite/vert_location_dectection.mp4', Fourcc, 25, (1440, 1080), False)
tracker = EuclideanDistTracker()
# mat=[]
DataFrame = pd.DataFrame(
    columns=['time', 'id', 'x', 'y', 'w', 'h', 'center_x', 'center_y'])

# 有两种算法可选,KNN和MOG2,下面的代码使用KNN作为尝试
backSub = cv.createBackgroundSubtractorKNN()


# find id def
def find_id(x, y, w, h, time, DataFrame, id_max,min_time):
    pos = np.array([x+w/2, y+h/2])
    # 从最近往前找
    dataframe_temp = DataFrame[DataFrame['time'] >=
                               min_time].sort_values(by='time', ascending=False)
    # print('len(df)',len(dataframe_temp))
    # print(dataframe_temp['time'].values)
    # 遍历所有可能的candidate
    for i in range(len(dataframe_temp)):
        temp_x, temp_y, temp_time, temp_id = dataframe_temp.iloc[i][[
            'center_x', 'center_y', 'time', 'id']].values
        temp_pos = np.array([temp_x, temp_y])
        #print('time',time,temp_pos,pos,np.linalg.norm(temp_pos-pos),(temp_time-time),(temp_time-time)*30)
        # when found
        if np.linalg.norm(temp_pos-pos) < 10:
            cv.putText(frame, "find" + str(temp_id), (x, y - 15), cv.FONT_ITALIC, 0.7, (255, 255, 255), 2)
            DataFrame.loc[len(DataFrame)]=[time, temp_id, x, y , w, h ,w/2+x, h/2+y]
            return(temp_id,DataFrame)
    # 遍历后还是找不到
    DataFrame.loc[len(DataFrame)]=[time, id_max+1, x, y , w, h ,w/2+x, h/2+y]
    cv.putText(frame, "new" + str(id_max+1), (x, y - 15), cv.FONT_ITALIC, 0.7, (255, 255, 255), 2)
    return(id_max+1,DataFrame)


# 打开一个视频文件
capture = cv.VideoCapture(cv.samples.findFileOrKeep('resource/vert.mov'))
# 判断视频是否读取成功
if not capture.isOpened():
    print('Unable to open')
    exit(0)


time = -1
id_max = 0
# 逐帧读取视频,进行相关分析
while (True):
    time += 1
    min_time = max(0, time-100)
    # 读取视频的第一帧
    ret, frame = capture.read()
    if frame is None:
        break
    # 使用定义的backSub对象,输入新的一帧frame,生成背景蒙版
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # frame=cv.GaussianBlur(frame,(5,5),2)
    # frame=np.where(frame[:,:,2]>110,255,0)

    fgMask = backSub.apply(frame)

    # 将原视频的当前帧和蒙版做相加运算,将前景物体提取出来
    Object = cv.add(frame, frame, mask=fgMask)
    Object = cv.blur(Object, (5, 5))
    _, Object = cv.threshold(Object, 80, 255, cv.THRESH_BINARY)
    Object = cv.blur(Object, (5, 5))
    #_ ,Object = cv.threshold(Object, 80, 255, cv.THRESH_BINARY)
    # Object=cv.erode(Object,None,iterations=1)
    # Object=cv.dilate(Object,None,iterations=1)

    contours, _ = cv.findContours(
        Object, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # 找到视频中物体的轮廓

    detections = []  # 用于存放boundingbox的起始点坐标.宽.高
    for i, cnt in enumerate(contours):
        # 计算出每个轮廓内部的面积,并根据面积的大小去除那些不必要的噪声（比如树.草等等）
        area = cv.contourArea(cnt)
        if area > 80 and area < 300:
            # cv.drawContours(Object, [cnt], -1, (0, 255, 0), 2)  # 画出移动物体的轮廓
            x, y, w, h = cv.boundingRect(cnt)
            cv.rectangle(frame, (x, y), (x + w, y + h),
                         (255, 255, 255), 2)  # 画出移动物体的外接矩形
            id,DataFrame=find_id(x, y, w, h, time, DataFrame, id_max,min_time)
            if id>id_max:
                id_max+=1
            DataFrame.append({
            'time': time,
            'id': id,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'center_x': int(x + w/2),
            'center_y': int(y + h/2)
            },ignore_index=True)


    cv.imshow('Frame', frame)
    #cv.imshow('FG Mask', fgMask)
    # cv.imshow('Object',Object)

    # 将当前帧写入输出文件
    frame.dtype = np.uint8
    out.write(frame)
    # mat.append(boxer_ids)
    # mat_1.append(detections[:,:2]+0.5*detections[:,2:])

    # 每帧展示结束,等待30毫秒
    keyboard = cv.waitKey(30)
    # 按q推出程序
    if keyboard == 27:
        break


# 存入npy文件
# mat=np.array(mat,dtype=object)
# np.save('result/boite/location_dectection.npy',mat)
DataFrame.to_csv('result/boite/vert_location_dectection.csv',
                 index=False, sep=',')

# 释放资源
capture.release()
out.release()
cv.destroyAllWindows()
