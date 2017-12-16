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