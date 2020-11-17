file1=open("F:\\pycharmproject\\GraduationProject\\data\\feature_data_new_statistic.txt",mode='r')
file2=open("F:\\pycharmproject\\GraduationProject\\data\\feature_data_new_statistic_part.csv",mode='a')
i=0
# for line in file1.readlines():
#     newline='0,'+line
#     file2.write(newline)
for line in file1.readlines():
    if i <=1987:
        file2.write(line)
    if i>=17000:
        file2.write(line)
    i+=1
