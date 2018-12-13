import numpy as np

[X_max,X_min,X_pixel,Y_max,Y_min,Y_pixel] = np.load('newdata/XYmaxmin1000.npy')

count=434730

if 1:
    X=np.zeros(count,float)
    Y=np.zeros(count,float)
    P=np.zeros(count,int)
    
    count=0
    
    file_object = open('newdata/poi_beijing_tencent.txt','rU')
    
    try: 
        for line in file_object:
            print(count)
            #print(line)
            line=line.strip('\n')
            line=line.split(' ', 2)
    
            X[count] = float(line[0])
            Y[count] = float(line[1])
            P[count] = int(line[2])
            count = count + 1
    finally:
         file_object.close()
         
    find_99=(P==99)
    P[find_99]=17
 
    #POI及其坐标
    np.save('newdata/XYP.npy',[X,Y,P])


[X,Y,P]=np.load('newdata/XYP.npy')

X=np.round((X-X_min)*(X_pixel-1)/(X_max-X_min)+0.5)
X=X.astype(int)
Y=np.round((Y-Y_min)*(Y_pixel-1)/(Y_max-Y_min)+0.5)
Y=Y.astype(int)
P=P.astype(int)

'''
#统计一共有几类POI  {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 99}
POI_class=set()
for i in range(count):
    POI_class.add(P[i])
'''

POI_number=17

POI=np.zeros([POI_number,int(X_pixel),int(Y_pixel)])

leave_out=0
for i in range(count):
    x=X[i]
    y=Y[i]
    p=P[i]-1
    if (x>=0 and x<=X_pixel-1) and (y>=0 and y<=Y_pixel-1):
        POI[p,x,y] = POI[p,x,y] + 1
    else:
        leave_out = leave_out + 1
print('leave_out=',leave_out)
poi=POI

np.save('newdata/P1000.npy',poi)


