file1=open("F:\\pycharmproject\\GraduationProject\\data\\feature_data_statistic.txt",mode='r')
file2=open("F:\\pycharmproject\\GraduationProject\\data\\feature_data_statistic_csv.csv",mode='w')
for f in file1.readlines():
    f=f.replace(" ",',')
    file2.write(f)
