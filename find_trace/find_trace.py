import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib as mpl

## include source
show_fourmis=pd.read_csv('result/boite/location_dectection_diplay.csv',dtype=np.int32)
print('total ant detected:',len(show_fourmis['id'].unique()))
#定义输出文件
Fourcc = cv.VideoWriter_fourcc('m','p','4','v')
out = cv.VideoWriter('result/boite/track_new.mp4',Fourcc, 25, (1440,1080),True)

#打开一个视频文件
capture = cv.VideoCapture(cv.samples.findFileOrKeep('resource/video_boite_entiere-test.ts'))
#判断视频是否读取成功
if not capture.isOpened():
    print('Unable to open')
    exit(0)

colors=[(128,128,128),(255,0,0),(0,0,255),(0,255,255),(255,255,0),(255,0,255),(118,128,105),(255,215,0),(176,23,31)]   
len_time=0



#逐帧读取视频,进行相关分析
while (True):
    #读取视频的第一帧
    len_time+=1
    ret, frame = capture.read()
    if frame is None:
        break
    #使用定义的backSub对象,输入新的一帧frame,生成背景蒙版
    #frame=cv.cvtColor(frame,cv.COLOR_BGR2GRAY) 
    if len_time in show_fourmis['time'].unique():
        
        
        df=show_fourmis[show_fourmis['time']<=len_time][['id','center_x','center_y']]
        x, y, w, h=show_fourmis[show_fourmis['time']==len_time]['x'].values,show_fourmis[show_fourmis['time']==len_time]['y'].values,show_fourmis[show_fourmis['time']==len_time]['w'].values,show_fourmis[show_fourmis['time']==len_time]['h'].values
        ids=show_fourmis[show_fourmis['time']==len_time]['id'].values

        cv.putText(frame, 'Total Ants detected in this frame:  '+str(ids.shape[0]), (20,20), cv.FONT_ITALIC, 0.7, (255, 255, 255), 2)
        id_set=np.sort(np.unique(ids))
        cv.putText(frame, 'Obj ids:  '+str(id_set), (50,50), cv.FONT_ITALIC, 0.7, (255, 255, 255), 2)

        for i in range(df.shape[0]):
            center_x=df['center_x'].values.astype(int)
            center_y=df['center_y'].values.astype(int)
            cv.circle(frame, (int(center_x[i]), int(center_y[i])), 1, (200,200,200), 2)
    
        for fourmi_id in range (ids.shape[0]):

                cv.putText(frame, 'Obj'+ str(ids[fourmi_id]), (x[fourmi_id], y[fourmi_id] - 15), cv.FONT_ITALIC, 0.7, colors[ids[fourmi_id]%len(colors)], 2)
                cv.rectangle(frame, (x[fourmi_id], y[fourmi_id]), (x[fourmi_id] + w[fourmi_id], y[fourmi_id] + h[fourmi_id]), colors[ids[fourmi_id]%len(colors)], 2)
                df_fourmis=df[df['id']==ids[fourmi_id]]
                if df.shape[0]>1:
                    center_x=df_fourmis['center_x'].values.astype(int)
                    center_y=df_fourmis['center_y'].values.astype(int)
                    for i in range(center_x.shape[0]):
                        cv.circle(frame, (int(center_x[i]), int(center_y[i])), 1, colors[ids[fourmi_id]%len(colors)], 2)
        

        
  

    


    #每帧展示结束,等待30毫秒
    keyboard = cv.waitKey(1)
    #按q推出程序
    if  keyboard == 27:
        break

    #将当前帧写入输出文件
    cv.imshow('Frame', frame)
    frame.dtype=np.uint8
    out.write(frame)

#释放资源
capture.release()
out.release()
cv.destroyAllWindows()


