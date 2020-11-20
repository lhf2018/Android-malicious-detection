# 把bw,mw改为0，1

file1 = open("H:\\A数据集\\others\\Features_file_csv_OmniDroid_v2\\features_file_ml7.csv", mode='r')
file2 = open("H:\\A数据集\\others\\Features_file_csv_OmniDroid_v2\\features_file_ml7_generate.csv", mode='a')

for line in file1.readlines():
    num1 = line.find(",")
    if line[0:num1]=='BW':
        file2.write('-1,' + line[num1 + 1:])
    else:
        file2.write('1,' + line[num1 + 1:])