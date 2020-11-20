import datetime
starttime = datetime.datetime.now()
#long running
for i in range(1,100000):
    continue
#do something other
endtime = datetime.datetime.now()
print (endtime - starttime)