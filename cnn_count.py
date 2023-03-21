import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
import pandas as pd
import cv2 as cv

pic_one = cv.imread('extract/nid/600.jpg')

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