# 本部分代码负责strace文件中提取特征向量所需的系统调用数量

file_strace = open("F:\\pycharmproject\\GraduationProject\\data\\adb_server_log.txt", mode="r")
map = dict()
for line in file_strace.readlines():
    num = line.find("(")
    if num > 0:
        sys_str = line[0:num]
        if sys_str in map.keys():
            map[sys_str] = map.get(sys_str) + 1
        else:
            map[sys_str] = 1
for key in map.keys():
    print(key + ": " + str(map[key]))
