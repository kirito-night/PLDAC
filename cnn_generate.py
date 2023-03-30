import cv2
import numpy as np
 
# 图片路径
 
#save y
img_list=[]
a = []
y_list=[]

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        a.append(x)
        xy = "%d,%d" % (x, y)
        cv2.circle(img_temp, (x, y), 5, (0, 0, 255), thickness=-1)
        #cv2.putText(img_temp, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,1.0, (255, 255, 255), thickness=1)
        cv2.imshow("image", img_temp)

# 图片路径
str_list=[str(i) for i in range(100,1400)] 
for s in str_list:
    print(s)
    img = cv2.imread('extract/nid/{}.jpg'.format(s))
<<<<<<< HEAD
    for i in range(10):
        for j in range(14):
            img_temp=img[i*100:(i+1)*100,j*100:(j+1)*100]
            img_list.append(img[i*100:(i+1)*100,j*100:(j+1)*100])
            a = []
            cv2.namedWindow("image")
            cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
            cv2.imshow("image", img_temp)
            keyboard = cv2.waitKey(0)
            b=len(a)
            #print(b)
            if keyboard=='q':
                cv2.destroyAllWindows()
                exit()
            y_list.append(b)
# X=np.load('X_small_train.npy').astype(np.float32)
# y_list=np.load('Y_small_train.npy')
# Y_0=np.where(y_list==0)[0]
# for i in Y_0:
#     cv2.imshow("image", X[i])
#     cv2.waitKey(250)
#     index=int(input(""))
#     y_list[i]=index
=======
    #for i in range(10):
        #for j in range(14):
    i,j=6,2
    img_temp=img[i*100:(i+1)*100,j*100:(j+1)*100]
    img_list.append(img[i*100:(i+1)*100,j*100:(j+1)*100])
    a = []
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    cv2.imshow("image", img_temp)
    keyboard = cv2.waitKey(0)
    b=len(a)
    #print(b)
    if keyboard=='q':
        cv2.destroyAllWindows()
        exit()
    y_list.append(b)

>>>>>>> e177557c560a7d91915c9313bff3795875bded0b

img_list=np.array(img_list,dtype=np.float32)
y_list=np.array( y_list,dtype=np.float32)
np.save('X_small_train{}.npy'.format(str(i)+str(j)),img_list)
np.save('Y_small_train{}.npy'.format(str(i)+str(j)), y_list)


cv2.destroyAllWindows()
# def rotate_image(image, angle):
#   image_center = tuple(np.array(image.shape[1::-1]) / 2)
#   rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
#   result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
#   return result

# def img_generate(img,XY):

#     [angle1, angle2,angle3] = np.random.choice([0, 90, 180, 270], 3)
#     img_copy=img.copy()
#     XY_copy=XY.copy()

#     img_piece1=img_copy[800:900,200:300]
#     img_piece2=img_copy[600:800,600:800]
#     img_piece3=img_copy[800:900,600:700]
#     XY_piece1=XY_copy[800:900,200:300]
#     XY_piece2=XY_copy[600:800,600:800]
#     XY_piece3=XY_copy[800:900,600:700]

#     img_copy[800:900,200:300]=rotate_image(img_piece1, angle1)
#     img_copy[600:800,600:800]=rotate_image(img_piece2, angle2)
#     img_copy[800:900,600:700]=rotate_image(img_piece3, angle3)

#     XY_copy[800:900,200:300]=rotate_image(XY_piece1, angle1)
#     XY_copy[600:800,600:800]=rotate_image(XY_piece2, angle2)
#     XY_copy[800:900,600:700]=rotate_image(XY_piece3, angle3)

#     return img_copy, XY_copy

# X_train=[]
# Y_train=[]
# str_list=[str(100*i) for i in range(1,15)]
# for num_str in  str_list:
#     img = cv2.imread('extract/nid/{}.jpg'.format(num_str))
#     XY=np.zeros(img.shape)
#     x=np.load('result/nid/x_{}.npy'.format(num_str))
#     y=np.load('result/nid/y_{}.npy'.format(num_str))
#     for i in range(len(x)):
#         XY[y[i],x[i],:]=1
#     for j in range(30):
#         img_result,XY_result=img_generate(img,XY)
#         X_train.append(img_result)
#         Y_train.append(XY_result)

# X_train=np.array(X_train)
# Y_train=np.array(Y_train)
# np.save("extract/nid/X_train.npy",X_train)
# np.save('extract/nid/Y_train.npy',Y_train)