# 生成特征向量,调用为1,未调用为0,正常为1,异常为2
import os
import pandas as pd
import chardet
import shutil

def move_normal_file(file_1):
    shutil.move("E:\\研究生\\毕业设计\\安卓恶意软件数据集\\Strace_OmniDroid_V2\\新建文件夹\\normal\\" + file_1
                , "E:\\研究生\\毕业设计\\安卓恶意软件数据集\\Strace_OmniDroid_V2\\新建文件夹\\used\\normal\\" + file_1)


def move_abnormal_file(file_2):
    shutil.move("E:\\研究生\\毕业设计\\安卓恶意软件数据集\\Strace_OmniDroid_V2\\新建文件夹\\abnormal\\" + file_2
                , "E:\\研究生\\毕业设计\\安卓恶意软件数据集\\Strace_OmniDroid_V2\\新建文件夹\\used\\abnormal\\" + file_2)




file_feature=open("F:\\pycharmproject\\GraduationProject\\data\\feature1.txt",mode='r')
file_data=open("F:\\pycharmproject\\GraduationProject\\data\\feature_data.txt",mode='a')
list=dict()
i=0
for line in file_feature.readlines():
    list[line.strip()]=i
    i+=1

path="E:\\研究生\\毕业设计\\安卓恶意软件数据集\\Strace_OmniDroid_V2\\新建文件夹\\abnormal"
files = os.listdir(path)
for file1 in files:
    # file_normal = open("E:\\研究生\\毕业设计\\安卓恶意软件数据集\\Strace_OmniDroid_V2\\新建文件夹\\normal\\" + file1,"rb")
    # print(chardet.detect(file_normal.read()))
    df=pd.read_csv("E:\\研究生\\毕业设计\\安卓恶意软件数据集\\Strace_OmniDroid_V2\\新建文件夹\\abnormal\\"+file1
                   ,delimiter="\t", encoding = 'ISO-8859-1', engine='python')
    # file_normal = open("E:\\研究生\\毕业设计\\安卓恶意软件数据集\\Strace_OmniDroid_V2\\新建文件夹\\normal\\"+file1
    #                    , mode='r',encoding='GBK')
    file_normal=df.values
    # 向量列表
    statistics=[0 for n in range(191)]
    # 处理每个文件中的向量
    for line1 in file_normal:
        temp=str(line1[0]).split(",")
        ll=temp[2]
        num1=ll.find("(")
        num2=ll.find("<")
        num3=ll.find("{")
        num4=ll.find("+")
        if(num2>-1 or num3 >-1 or num4>-1):
            continue
        # print(list.get(ll[:num1]))
        # print(ll[:num1])
        if(list.get(ll[:num1])==None):
            print(ll[:num1])
        statistics[list.get(ll[:num1])]=1
        # print(ll[:num1])
    s = ""
    for item in statistics:
        s += str(item) + " "
    s += "2"
    s+='\n'
    file_data.write(s)
    move_abnormal_file(file1)