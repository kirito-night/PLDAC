import cv2
from track import *

# 进行追踪
tracker = EuclideanDistTracker()  # 这个函数通过获取同一物体不同时刻的boundingbox的坐标从而实现对其的追踪

# 导入想要进行tracking的视频，要求拍摄视频的过程中摄像头是保持静止状态的
cap = cv2.VideoCapture("resource/video_juste_nid-test.ts")

# 从导入的视频中找到正在移动的物体
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)  # 对参数进行调整则会改变捕捉移动物体的精准性

while True:
    ret, frame = cap.read()
    height, width, _ =frame.shape  # 得出视频画面的大小，从而去方便计算出感兴趣区域所在的位置
    print(height, width)  # 720,1280

    # 设置一个感兴趣区域，让处理（对物体的detection和tracking）只关注于感兴趣区域，从而减少一些计算量也让检测变得简单一些
    roi = frame

    # 物体检测  （根据需要可以将该部分代码换成比如行人检测、汽车检测等等）
    mask = object_detector.apply(roi)  # 通过加一个蒙版，更加清晰的显示出移动中的物体，即只留下白色的移动的物体。
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)  # 去除移动物体被检测到的时候所附带的阴影（阴影为灰色）
    #将原视频的当前帧和蒙版做相加运算，将前景物体提取出来
    Object=cv2.add(frame,frame,mask=mask)

    # Grayscale
    gray = cv2.cvtColor(Object, cv2.COLOR_BGR2GRAY)
    # Find Canny edges
    edged = cv2.Canny(gray, 30, 200)
 
    # Finding Contours
    # Use a copy of the image e.g. edged.copy()
    # since findContours alters the image
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找到视频中物体的轮廓

    detections = []  # 用于存放boundingbox的起始点坐标、宽、高
    for cnt in contours:
        # 计算出每个轮廓内部的面积，并根据面积的大小去除那些不必要的噪声（比如树、草等等）
        area = cv2.contourArea(cnt)
        if area > 100:
            # cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)  # 画出移动物体的轮廓
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])


    # 物体追踪
    boxer_ids = tracker.update(detections)  # 同一个物体会有相同的ID
    # print(boxer_ids)
    for box_id in boxer_ids:
        x, y, w, h, id = box_id
        cv2.putText(roi, "Obj" + str(id), (x, y - 15), cv2.FONT_ITALIC, 0.7, (255, 0, 0), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 根据移动物体的轮廓添加boundingbox

    #print(detections)
    cv2.imshow("Frame", frame)  # 打印结果
    # cv2.imshow("Mask", mask)  # 打印出蒙版
    # cv2.imshow("ROI", roi)  # 打印出你想要的ROI在哪
    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
