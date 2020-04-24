# 本部分代码负责logcat文件中提取特征向量所需的安卓调用数量

file_strace = open("F:\\pycharmproject\\GraduationProject\\data\\logcat.txt"
                   , mode="r",encoding='UTF-8')
map = dict()
for line in file_strace.readlines():
    num1 = line.find("/")
    num2 = line.find("(")

    if num1 > 0:
        sys_str = line[num1+1:num2].strip()
        if sys_str in map.keys():
            map[sys_str] = map.get(sys_str) + 1
        else:
            map[sys_str] = 1
# 把特征写入文件中
file_feature=open("F:\\pycharmproject\\GraduationProject\\data\\feature.txt",'a')
for key in map.keys():
    print(key + ": " + str(map[key]))
    file_feature.writelines(key+'\n')
