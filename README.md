# BeautifulGirls
写真美女套图：爬虫+美女脸部识别+DCGAN脸部自动生成

所有具体的内容信息，代码解析，步骤分析请到我的博客观看具体教程。

http://blog.csdn.net/sileixinhua/article/details/78816683

谢谢。

# 写真美女套图：爬虫+美女脸部识别+DCGAN脸部自动生成


**所有代码请到我的github上下载，请star一下，谢谢大家了。**

https://github.com/sileixinhua/BeautifulGirls

# 第一部分：爬虫 抓美女套图（Python+BeautifulSoup+requests）

## 前言

本文主要是以爬虫爬取下来的图片为数据，做一个只针对美女脸部识别，和一个DCGAN合成美女脸的模型。

第一部分：写爬虫主要看需求来决定工具的使用，python无非是众多语言中比较成熟的一个，如果要分析json，要分布式，就用scrapy，如果功能要求简单的就用BeautifulSoup+requests就可以了。requests用于和服务器的交互，BeautifulSoup解析HTML页面格式数据，并提取想要的信息。

第二部分：现在脸部，物体的识别多是用tensorflow等机器学习框架来做，但是其实在很早的时候用opencv就可以做了，现在opencv也有DNN等功能，这里美女脸部识别主要是用了opencv的cascades识别功能，这部分基本不要写什么代码，但是过程会比较繁琐。

第三部分：有人说近十年深度学习的重要发现就是GAN，第三部分就是用了GAN+CNN的变种DCGAN，GAN主要是用来根据现有的数据发现其中的模式生成数据，图像，语音等，主要组成部分为生成器和判别器，生成器是原始数据加噪声来合成新的数据，判别器主要是根据原始数据判别生成数据的相似度，准确度。


## 开发环境

**windows10**

**Python3.5**

https://www.python.org/downloads/

![这里写图片描述](http://img.blog.csdn.net/20171215205629828?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2lsZWl4aW5odWE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

**BeautifulSoup**

https://www.crummy.com/software/BeautifulSoup/bs4/doc/index.zh.html

![这里写图片描述](http://img.blog.csdn.net/20171215205738236?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2lsZWl4aW5odWE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

**Requests**

http://docs.python-requests.org/en/master/#

![这里写图片描述](http://img.blog.csdn.net/20171215205655705?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2lsZWl4aW5odWE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

可能需要的python包安装（Python3环境）

```

pip3 install BeautifulSoup

```


```
pip3 install requests

```


```

pip3 install lxml
```

这里还是推荐使用Python3，但是用Python2的同学，把上述命令的“pip3”改成“pip”就可以了。

## 爬虫目标网页结构分析

目标网址：http://www.xingmeng365.com/

爬虫需要抓取的页面：

![这里写图片描述](http://img.blog.csdn.net/20171215213024413?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2lsZWl4aW5odWE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![这里写图片描述](http://img.blog.csdn.net/20171215213428326?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2lsZWl4aW5odWE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

爬虫需要抓取的页面地址：

![这里写图片描述](http://img.blog.csdn.net/20171215210931589?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2lsZWl4aW5odWE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

爬虫获取HTML页面信息的地址：

![这里写图片描述](http://img.blog.csdn.net/20171215211044093?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2lsZWl4aW5odWE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

## 代码分析

SpiderDownloadImages.py

```
# 2017年11月10日 19点24分
# 作者：橘子派_司磊
# 爬虫：抓美女套图
# 目标网址：http://www.xingmeng365.com/

from bs4 import BeautifulSoup
import requests
import os
import urllib.request

# 在mian.py当前位置创建图片收集的文件夹Photos
if not os.path.exists('Photos'):
		os.makedirs('Photos')

num = 67
image_list = []
id = 7

while(id <= 559):
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0'}
    url = requests.get('http://www.xingmeng365.com/articles.asp?id='+str(id), headers=headers)
    # 此处用 “，str(id)” 的话，逗号打印出来会变成 “id=&2”
    print("当前爬取的网址为："+url.url)
    html_doc = url.text
    # 此处用url不带".text"的话报错，Python: object of type 'Response' has no len()
    # 错误解决
    # https://stackoverflow.com/questions/36709165/python-object-of-type-response-has-no-len
    soup = BeautifulSoup(html_doc,"lxml")

    for link in soup.find_all('img'):
        if "/upload" in link.get('src'):
            # id=7以后，"../../"改为"/upload"
            image_url = link.get('src')
            # 获得的图片地址有错，需要改成
            # http://www.xingmeng365.com/upload/image/20170811/20170811203590079007.jpg
            # 即把 “../../” 改为 “http://www.xingmeng365.com/”
            # id=7 以后为/upload/image/20170811/20170811210596789678.jpg
            # 即http://www.xingmeng365.com/upload/image/20170811/20170811210545754575.jpg
            image_url = "http://www.xingmeng365.com/" + image_url[1:]
            # id=7以后，[6:]改为[1:]
            print("开始下载第"+str(num+1)+"张图片："+image_url)
            file = open('Photos/'+str(num)+'.jpg',"wb")
            req = urllib.request.Request(url=image_url, headers=headers) 
            try:
                image = urllib.request.urlopen(req, timeout=10)
                pic = image.read()
            except Exception as e:
                print("第"+str(num+1)+"张图片访问超时，下载失败："+image_url)
                continue
            # 遇到错误，网站反爬虫
            # urllib.error.HTTPError: HTTP Error 403: Forbidden
            # 原因是这里urllib.request方法还需要加入“, headers=headers”
            # 头文件来欺骗，以为我们是客户端访问
            file.write(pic)
            print("第"+str(num+1)+"张图片下载成功")
            file.close()
            num = num + 1
    id = id + 1
```

## 实验结果

视网络情况而定，一共花费7293.5秒，爬取15684张图。

![这里写图片描述](http://img.blog.csdn.net/20171216093436605?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2lsZWl4aW5odWE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


![这里写图片描述](http://img.blog.csdn.net/20171215210057137?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2lsZWl4aW5odWE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


![这里写图片描述](http://img.blog.csdn.net/20171215210114947?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2lsZWl4aW5odWE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

这里文件夹下所有的图像数据就保存在Photos文件夹里，一共有15684张图。

# 第二部分 美女脸部识别（Python+Opencv的Cascades）

## 前言

现在大家都在用TensorFlow等神经网络框架做识别，过程繁琐，有些功能可以直接用OpenCV做到，而且封装好的开发工具包可以节省很多时间，效果也还可以。

这里解释一下OpenCV自带的cascades识别，官网信息地址。

https://docs.opencv.org/3.0.0/d7/d8b/tutorial_py_face_detection.html

首先安装OpenCV后，源码目录下\opencv\sources\data\haarcascades，就有很多自带的人脸识别.xml
文件，这个文件里包含的就是要识别出物体的信息。

## 开发环境

Python3 + OpenCV

OpenCV的window安装直接官网https://opencv.org/下载源码，把bin路径添加到系如变量即可

在Ubuntu上的安装比较繁琐，我找到的最简单的方式是：

https://www.youtube.com/watch?v=2Pboq2LFoaI

http://www.daslhub.org/unlv/wiki/doku.php?id=opencv_install_ubuntu

整个过程安装比较耗时，大概一刻钟左右。

在Python中安装OpenCV开发包需要如下命令：

```
pip3 install opencv-python
```

这里如果是Python2就把“pip3”改成“pip”即可。

## 实验分析与步骤设计

本实验步骤有点繁琐，请仔细查阅，经过第一部分，文件下已经有Photos文件夹，这里是所有美女写真套图的数据集。

 1. 我们需要用OpenCV自带的脸部识别把所有美女套图的脸部截取下来，存放进Faces文件夹里。
 2. 然后用OpenCV自己的方法创建我们自己的cascades的识别器，用来识别美女的脸部，丑的不识别，这一步主要就是生成.xml文件，文件里包含的就是美女脸部的信息。Negative文件夹里是背景，即负面Negative数据，用来和真实的脸部数据做对比，让训练器知道哪些是美女脸，哪些不是。结果会生成进data文件夹里，结果是一个.xml类型的文件，具体步骤如下代码分析所示。
 3. 用我们自己生成的.xml文件来识别美女的脸部，丑的不识别。 

这一部分的实验主要是OpenCV的脸部识别器我用错了，所以从截取的脸部信息就有噪声数据，即不是脸部的图也被截取下来混进去了，所以效果不是很好，如果要提高效果，可以用别的脸部识别分类器，或者手动删除非脸部图片，并加大Negative文件夹里的图片即可。

## 代码分析

首先是上面实验分析的第一步，截图美女脸部图像

TakeImgFace.py

```
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
```

这步完成之后，在faces文件夹里会有很多很多的美女脸部的截图，但是识别器可以用更好的，我这里只做示范，有些噪音数据就先放着不管了，你们自己做的时候想提高效果可以手动去除噪音数据，或者用更好的分类器。

![这里写图片描述](http://img.blog.csdn.net/20171216101304938?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2lsZWl4aW5odWE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

\opencv\sources\data\haarcascades文件下有很多识别器，你们可以自己试试效果如果。用效果最好的那个即可。

接下来就是重难点了，用OpenCV的cascades创建我们自己的识别器。

首先需要获取正面和负面数据的数据列表信息。

CreateInfoTxt.py

```
import os

# 创建positive.txt和negative.txt文件
# 文件内容是数据集的list

def Create_faces_info_lst():
    for file_type in ['Faces']:
    
        for img in os.listdir(file_type):

            if file_type == 'Faces':
                line = file_type+'/'+img+' 1 0 0 48 48\n'
                with open('info.lst','a') as f:
                    f.write(line)

if __name__=="__main__":
    Create_faces_info_lst()
```

上面的代码运行之后会生成一个info.lst文件，里面会有Faces正面数据里的数据列表（让OpenCV知道你一共有多少正面数据有哪些），然后下载去我的github里下载背景即负面数据的数据集，https://github.com/sileixinhua

info.lst文件如下所示：

![这里写图片描述](http://img.blog.csdn.net/20171216101343395?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2lsZWl4aW5odWE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

负面Negative文件夹数据如下所示：

![这里写图片描述](http://img.blog.csdn.net/20171216101432275?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2lsZWl4aW5odWE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

这个时候有会用到Faces文件夹下美女脸部数据，info.lst正面数据列表信息（让OpenCV知道你一共有多少正面数据有哪些），从我github上下载的Negative文件夹，接下在本地项目文件夹下打开cmd。linux的打开terminal。运行如下命令

```
opencv_createsamples -info info.lst -num 14229 -w 48 -h 48 -vec positive.vec
```
这一行命令是根据正面数据的信息创建positive.vec文件，用来告诉opencv正面数据的特征。

如果报错"Parameters can not be written, because file data/params.xml can not be opened."，请在项目文件夹里创建一个名字为“data”的文件夹，

```
opencv_traincascade -data data -vec positive.vec -bg bg.txt -numPos 12000 -numNeg 202 -numStages 20 -w 48 -h 48
```
这一行命令就是训练我们的cascades识别器了，存放地址为data文件夹，vec就是我们上一步创建的positive.vec，bg就是Negative文件夹里负面数据的列表，训练区块为20个，根据数据集的大小可以调整，输入为高48，宽48。这里我做错了，应该填96。（这里一错又导致我效果不好），这里我一共花了2天采用CPU计算完毕。

全部计算结束后请再次输入一次上面的命令

```
opencv_traincascade -data data -vec positive.vec -bg bg.txt -numPos 12000 -numNeg 202 -numStages 20 -w 48 -h 48
```

用来把每一个区块的.xml结果信息合成为一个.xml文件

然后把data文件夹里生成的.xml文件改为你想要的名字，我就改为BeautifulFacaCascade.xml。

下一步就是第二部分的最后一部，识别美女脸部，我记录了视频和图片两种，代码如下所示：

CascadaBeautifulFace.py

```
import cv2
import os
import numpy as np

# opencv_createsamples -info info.lst -num 14229 -w 48 -h 48 -vec positive.vec

# create data file 
# or will error "Parameters can not be written, because file data/params.xml can not be opened."

# opencv_traincascade -data data -vec positive.vec -bg bg.txt -numPos 12000 -numNeg 202 -numStages 20 -w 48 -h 48

# opencv_traincascade -data data -vec positive.vec -bg bg.txt -numPos 12000 -numNeg 202 -numStages 20 -w 48 -h 48

# ----------------------------------------------------------------------------------------------
# use video

# beautiful_face_cascade = cv2.CascadeClassifier('C:\\Code\\BeautifulGirls\\BeautifulFacaCascade.xml')

# cap = cv2.VideoCapture(0)

# while 1:
#     ret, img = cap.read()

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     beautiful_face = beautiful_face_cascade.detectMultiScale(gray, 1.3, 5)
#     # 这里参数可改成 5
#     # detectMultiScale()
#     # https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html
#     # minSize – Minimum possible object size. Objects smaller than that are ignored.
#     # maxSize – Maximum possible object size. Objects larger than that are ignored.
    
#     for (x,y,w,h) in beautiful_face:
#         cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)

#     cv2.imshow('img',img)
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()

# ---------------------------------------------------------------------------------------
# use image

BeautifulFacaCascade = cv2.CascadeClassifier('C:\\Code\\BeautifulGirls\\BeautifulFacaCascade.xml')
img = cv2.imread('0.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

BeautifulFaca = BeautifulFacaCascade.detectMultiScale(gray, 500, 500)
for (x,y,w,h) in BeautifulFaca:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 实验结果

![这里写图片描述](http://img.blog.csdn.net/20171216102951463?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2lsZWl4aW5odWE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

其实这一部分结果不是很好，原因有两个，一是脸部截取的数据有噪音，二是负面数据太少，但是我训练这些用了32G8核的7700K用了2天，应该改用GPU运算，最近事情比较多，我也就没再次训练了，有兴趣的同学可以自己再试一试。

第二部分十分繁琐，如果描述的不清楚，大家可以去网上搜索一下OpenCV Cascades，中文信息不多，最好用Google搜索。

# 第三部分 DCGAN脸部自动生成（Python+Tensorflow +DCGAN）

## 前言

## 开发环境

Python3 + Tensorflow

Tensorflow的安装在windows上十分繁琐，linux也一样。主要是cudnn和cuda的安装麻烦。

去YouTuBe上去找视频看的话，但是绝大部分都是一年两年以前的视频。

安装请按照官方网站的来https://www.tensorflow.org/versions/master/install/install_linux

但是windows和linux上不用gpu运算的话可以用cpu运算，安装就十分简单了，本文第三部分实验我就是在自己笔记本上cpu运算的，耗时3，4小时而已。

安装命令如下：

```
pip3 install tensorflow
```

Python2的改“pip3”为“pip”。

> 上周我实验室的电脑系统崩了，还好我都有备份。但是我重装ubuntu之后，发现现在tensorflow不支持最新的cuda9，结果现在cuda官网只有9，测试失败之后又重装的系统装cuda8配cudnn，大家以后装tensorflow的时候可以注意，网上的教程一天一遍，过几个月基本都没用，还是要按照官方的安装指导来。

## 实验分析与步骤设计

这部分实验都在文件夹DCGAN里。

首先在DCGAN文件夹下创建data文件夹，并把之前Faces文件夹复制到data文件夹里。

这里的代码我是直接在github上找到的代码，可以直接用。

https://github.com/carpedm20/DCGAN-tensorflow

## 代码分析

在DCGAN文件下打开cmd或者terminal，运行如下命令。

```
python main.py --input_height 96 --input_width 96 --output_height 48 --output_width 48 --dataset Faces --crop --train --epoch 300 --input_fname_pattern "*.jpg"
```

这个人的代码十分有用，推荐有兴趣的同学好好看看分析一下。但是看之前还是建议把“花书”看了。

## 实验结果

运行的效果如下图所示：

![这里写图片描述](http://img.blog.csdn.net/20171216104224962?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2lsZWl4aW5odWE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

实验结果如下所示：

![这里写图片描述](http://img.blog.csdn.net/20171216104350980?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2lsZWl4aW5odWE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![这里写图片描述](http://img.blog.csdn.net/20171216104409878?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2lsZWl4aW5odWE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

再训练下去只会更加清晰，有条件的同学可以把结果传群里，谢谢。
