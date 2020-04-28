file1=open("F:\\pycharmproject\\GraduationProject\\data\\feature_data - 副本.txt",mode='r')
file2=open("F:\\pycharmproject\\GraduationProject\\data\\feature_data - 副本1.txt",mode='w')
for f in file1.readlines():
    f=f.replace(" ",',')
    file2.write(f)
