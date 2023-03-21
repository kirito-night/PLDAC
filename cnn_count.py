import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
import pandas as pd
import cv2 as cv

pic_one = cv.imread('extract/nid/500.jpg')


pic_smalls=[]
for i in range((pic_one.shape[0])//400):
    for j in range((pic_one.shape[1])//400):
        pic_smalls.append(pic_one[i*400:(i+1)*400,j*400:(j+1)*400])

np.save('result/nid/x_500.npy',pic_smalls)
# pic_labs=[]
# for i in range(len(pic_smalls)):
#     #print(pic_smalls[i].shape)
#     cv.imshow('pic',pic_smalls[i])
#     keyboard = cv.waitKey(25)
#     #按q推出程序
#     if  keyboard == 27:
#         break
#     y=float(input('combien?'))
#     pic_labs.append(y)

# np.save('result/nid/y_1300.npy',pic_labs)
# cv.destroyAllWindows()