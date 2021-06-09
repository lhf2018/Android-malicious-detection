import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5, 6]
y_pre = [0.97215, 0.97784, 0.97946, 0.97733, 0.97730, 0.97676]
y_re = [0.97201, 0.97785, 0.97941, 0.97725, 0.97727, 0.97674]
y_f1 = [0.97189, 0.97784, 0.97945, 0.97729, 0.97729, 0.97675]
y_time = [38.891, 139.702, 199.553, 272.262, 354.704, 482.176]


# l1=plt.plot(x,y_pre,'r--',label='precision')
# l2=plt.plot(x,y_re,'g--',label='recall')
# l3=plt.plot(x,y_f1,'b--',label='f1-score')
l4=plt.plot(x,y_time,'y--',label='training time(s)')


# plt.plot(x,y_pre,'ro-',x,y_re,'g+-',x,y_f1,'b^-',x,y_time,'y*-')
# plt.plot(x,y_pre,'ro-',x,y_re,'g+-',x,y_f1,'b^-')
plt.xlabel('Number of layers')

plt.legend()
plt.show()
