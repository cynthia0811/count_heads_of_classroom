# 这个函数里的rect是dlib脸部区域检测的输出。这里将rect转换成一个序列，序列的内容是矩形区域的边界信息。
def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

# 这个函数里的shape是dlib脸部特征检测的输出，一个shape里包含了前面说到的脸部特征的68个点。这个函数将shape转换成Numpy array，为方便后续处理。
def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68,2),dtype=dtype)
    for i in range(0,68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

# 这个函数里的image就是我们要检测的图片。在人脸检测程序的最后，我们会显示检测的结果图片来验证，这里做resize是为了避免图片过大，超出屏幕范围。
def resize(image, width=1200):
    r = width *1.0/ image.shape[1]
    dim = (width,int(image.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

# 主程序
# 输出数组judge,表示每个区域的的人数
import sys
import numpy as np
import dlib
import cv2

if len(sys.argv)<2:
    print(len(sys.argv))
    print("Usage: %s <image file>"% sys.argv[0])
    sys.exit(1)
image_file = sys.argv[1]
detector = dlib.get_frontal_face_detector()

# 读入图像
image = cv2.imread(image_file)
#图像长宽
height,width = image.shape[:2]
# 分块
h = int(height/2)
w = int(width/2)
box=(0,0,h,w)
# 截取四个区域图片
image1 = image[0:h,0:w]
image2 = image[0:h,w:width]
image3 = image[h:height,0:w]
image4 = image[h:height,w:width]

judge = [0,0,0,0]
m = 0
for image in (image1,image2,image3,image4):
    # resize成合适的尺寸
    image = resize(image, width=1200)
    # 灰度化
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    rects = detector(gray,1)
    # 预测
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    x = 0
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        (x, y, w, h) = rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.putText(image,"Face #{}".format(i +1), (x -10, y -10),
        #         cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,255,0),2)  #标注人脸个数
        # for (x, y) in shape:cv2.circle(image,(x, y),2,(0,0,255),-1)#输出关键点
        # 人数计算
        if x > 0: judge[m] += 1
    m +=1

print("各个区域的人数为",judge)#判断人数
out_put = [1,1,1,1]
for i in range(4):
	if judge[i]>0:out_put[i]=0
print("电平输出(有人输出0，无人输出1)",out_put)#输出电平，有人为0，无人为1
