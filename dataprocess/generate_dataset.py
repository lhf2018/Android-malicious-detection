file1 = open("H:\\A数据集\\others\\Features_file_csv_OmniDroid_v2\\features_file_ml7_generate.csv", mode='r')
file2 = open("H:\\A数据集\\others\\Features_file_csv_OmniDroid_v2\\new3_features_file_ml7_generate.csv", mode='a')
i = 0
j = 0
for line in file1.readlines():
    ind = line.index(",")
    if line[0:ind] == "1" and i <= 1000:
        file2.write(line)
        i += 1
    elif line[0:ind] == "-1" and j <= 1000:
        file2.write(line)
        j += 1
