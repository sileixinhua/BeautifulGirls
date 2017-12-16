import cv2
import sys
import os.path
from glob import glob

# C:\Code\BeautifulGirls\Faces
# C:\Code\BeautifulGirls\Photos

# 一共有15683张写真美图

# 本文件是用来从Photos写真美图文件夹中，用opencv自带的人脸识cascade别出脸部并截图保存到Faces文件夹中

# 在opencv的自带人脸检测中，haarcascade_frontalface_alt效果最好，缺点是时间长

def detect(filename, cascade_file="C:\\OpenCV\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)
        # 这里确认找到cascades识别器，找不到显示not found，地址请根据你的自己安装位置修改一下

    cascade = cv2.CascadeClassifier(cascade_file)
    # 导入识别器
    image = cv2.imread(filename)
    # 读取图片
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 获取图片的灰度图
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(gray,
                                     scaleFactor=1.1,
                                     minNeighbors=5,
                                     minSize=(48, 48))
    # 识别脸部
    for i, (x, y, w, h) in enumerate(faces):
    	# 定义脸部在图像上的坐标
        face = image[y: y + h, x:x + w, :]
        # 获取坐标位置的图
        face = cv2.resize(face, (96, 96))
        # 重新定义大小
        save_filename = '%s-%d.jpg' % (os.path.basename(filename).split('.')[0], i)
        # 定义保存图片的地址
        cv2.imwrite("Faces/" + save_filename, face)
        # 保存图片


if __name__ == '__main__':
    if os.path.exists('Faces') is False:
        os.makedirs('Faces')
    # 检查Faces文件夹，没有就创建一个
    file_list = glob('Photos/*.jpg')
    for filename in file_list:
        detect(filename)