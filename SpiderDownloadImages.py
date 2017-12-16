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