# 生成特征向量,统计数量,正常为1,异常为2
import os
import pandas as pd
import chardet
import shutil

def move_normal_file(file_1):
    shutil.move("H:\\A数据集\\normal\\" + file_1
                , "H:\\A数据集\\used\\normal\\" + file_1)


def move_abnormal_file(file_2):
    shutil.move("H:\\A数据集\\abnormal\\" + file_2
                , "H:\\A数据集\\used\\abnormal\\" + file_2)




file_feature=open("F:\\pycharmproject\\GraduationProject\\data\\feature2.txt",mode='r')
file_data=open("F:\\pycharmproject\\GraduationProject\\data\\feature_data_new_statistic.txt",mode='a')
list=dict()
i=0
for line in file_feature.readlines():
    list[line.strip()]=i
    i+=1

path="H:\\A数据集\\abnormal"
files = os.listdir(path)
for file1 in files:
    df=pd.read_csv(path+"\\"+file1
                   ,delimiter="\t", encoding = 'ISO-8859-1', engine='python')
    file_normal=df.values
    # 向量列表
    statistics=[0 for n in range(192)]
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
        if(list.get(ll[:num1])==None):
            print(ll[:num1])
        statistics[list.get(ll[:num1])]+=1
    s = ""
    for item in statistics:
        s += str(item) + ","
    s += "2"
    s+='\n'
    file_data.write(s)
    move_abnormal_file(file1)