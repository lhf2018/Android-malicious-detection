# 统计系统调用变量

file=open("F:\\pycharmproject\\GraduationProject\\data\\feature.txt")
file2=open("F:\\pycharmproject\\GraduationProject\\data\\feature1.txt",mode='w')
for line in file.readlines():
    num=line.strip().split(" ")
    file2.write(num[1]+'\n')