import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
import pandas as pd
import cv2 as cv

y1=np.load('result/nid/y_1300.npy')
y2=np.load('result/nid/y_1200.npy')
print(y1+y2)