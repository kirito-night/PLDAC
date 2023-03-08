import cv2 as cv
from track import *
import numpy as np
import pandas as pd
## include source
show_fourmis=pd.read_csv('result/nid/location_dectection_diplay.csv',dtype=np.int32)

#定义输出文件
Fourcc = cv.VideoWriter_fourcc(*'MPEG')
out = cv.VideoWriter('result/boite/seul_points.ts',Fourcc, 20, (1440,1080),False)

#打开一个视频文件
capture = cv.VideoCapture(cv.samples.findFileOrKeep('resource/video_juste_nid-test.ts'))
#判断视频是否读取成功
if not capture.isOpened():
    print('Unable to open')
    exit(0)


len=0
#逐帧读取视频,进行相关分析
while (True):
    #读取视频的第一帧
    len+=1
    ret, frame = capture.read()
    if frame is None:
        break
    #使用定义的backSub对象,输入新的一帧frame,生成背景蒙版
    frame=cv.cvtColor(frame,cv.COLOR_BGR2GRAY) 
    if len in show_fourmis['time'].unique():
        print('frame=',len)
        center_x,center_y=show_fourmis[show_fourmis['time']<=len]['center_x'].values,show_fourmis[show_fourmis['time']<=len]['center_y'].values
        x, y, w, h=show_fourmis[show_fourmis['time']==len]['x'].values,show_fourmis[show_fourmis['time']==len]['y'].values,show_fourmis[show_fourmis['time']==len]['w'].values,show_fourmis[show_fourmis['time']==len]['h'].values
        ids=show_fourmis[show_fourmis['time']==len]['id'].values
        print('ids shape',ids.shape[0])
        print(ids[0])
        if ids.shape[0]>1:
            for fourmi_id in range (ids.shape[0]):
                cv.putText(frame, 'Obj'+ str(ids[fourmi_id]), (x[fourmi_id], y[fourmi_id] - 15), cv.FONT_ITALIC, 0.7, (255, 255, 255), 2)
                cv.rectangle(frame, (x[fourmi_id], y[fourmi_id]), (x[fourmi_id] + w[fourmi_id], y[fourmi_id] + h[fourmi_id]), (255, 255, 255), 2)
        for i in range(center_x.shape[0]):
            cv.circle(frame, (int(center_x[i]), int(center_y[i])), 1, (255, 255, 255), 2)

    #将当前帧写入输出文件
    cv.imshow('Frame', frame)
    out.write(frame)


    #每帧展示结束,等待30毫秒
    keyboard = cv.waitKey(30)
    #按q推出程序
    if  keyboard == 27:
        break



#释放资源
out.release()
cv.destroyAllWindows()


