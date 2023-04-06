import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import cv2 as cv


## include source
mu=np.load('recognition_img_fix/mu.npy')
sig=np.load('recognition_img_fix/sig.npy')

#定义输出文件
Fourcc = cv.VideoWriter_fourcc('m','p','4','v')
out = cv.VideoWriter('result/boite/nid_count.mp4',Fourcc, 25, (1440,1080),True)

# Init model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(4*4*32, 6)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
       
# Initialize the model
model_frac = CNN()
# Load the model
model_frac=torch.load('recognition_img_fix/model_frac.pkl')

# fonction for each img
def pred_pic(img_init,model_frac):
    img=img_init.copy()
    # read img
    len_i=img.shape[0]//64
    len_j=img.shape[1]//64  

    # transform and normalization(standarlization)
    XX_img=np.zeros((1, 3, img.shape[0],img.shape[1])).astype(np.float32)
    XX_img[0,0]=(img[:, :, 0]-mu[:,:,0])/sig[:,:,0]
    XX_img[0,1]=(img[:, :, 1]-mu[:,:,0])/sig[:,:,0]
    XX_img[0,2]=(img[:, :, 2]-mu[:,:,0])/sig[:,:,0]

    # break into small pieces
    img_frac=np.zeros((len_i*len_j,3,64,64))
    for i in range (len_i):
        for j in range(len_j):
            img_frac[j+i*len_j]=XX_img[:,:,64*i:64*(i+1),64*j:64*(j+1)]

    # evaluate by model
    model_frac.eval()
    with torch.no_grad():
        images=torch.from_numpy(img_frac).float()
        outputs = model_frac(images)
        _, predicted = torch.max(outputs.data, 1)
        #print(labels,predicted)
        #print('total ant count :  ', sum(predicted))
    
    # show the result
    for i in range(len_i):
        for j in range(len_j):
            cv.rectangle(img,(j*64, i*64), ((j+1) * 64, (i+1) * 64), (255, 255, 255), 2)
            cv.putText(img, str(predicted[j+i*len_j].item()), (j*64+30, i*64+30), cv.FONT_ITALIC, 1, (255, 255, 255), 2)
    cv.putText(img, 'total ant count :  '+str(sum(predicted).item()), ((j-20)*64, (i+1)*64+30), cv.FONT_ITALIC, 2, (255, 255, 255), 2)
    return (img,sum(predicted).item())

#打开一个视频文件
capture = cv.VideoCapture(cv.samples.findFileOrKeep('resource/video_juste_nid-test.ts'))
#判断视频是否读取成功
if not capture.isOpened():
    print('Unable to open')
    exit(0)

count=0
pred_old=0
#逐帧读取视频,进行相关分析
while (True):
    count+=1
    #读取视频的第一帧
    
    ret, frame = capture.read()
    len_i=frame.shape[0]//64
    len_j=frame.shape[1]//64  
    if frame is None:
        break
    #使用定义的backSub对象,输入新的一帧frame,生成背景蒙版
    #frame=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    img,pred=pred_pic(frame,model_frac)
    if abs(pred_old-pred)>10:
        pred_old=pred
    cv.putText(frame, 'total ant count :  '+str(pred_old), ((len_j-20)*64, len_i*64+30), cv.FONT_ITALIC, 2, (255, 255, 255), 2)
    cv.imshow('Frame', frame)
    cv.imshow('IMG', img)

    #每帧展示结束,等待30毫秒
    keyboard = cv.waitKey(30)
    #按q推出程序
    if  keyboard == 27:
        break

    #将当前帧写入输出文件

    frame.dtype=np.uint8
    out.write(img)

#释放资源
capture.release()
out.release()
cv.destroyAllWindows()


