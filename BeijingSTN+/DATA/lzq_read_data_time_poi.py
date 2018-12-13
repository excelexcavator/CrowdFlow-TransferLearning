import numpy as np

class MM:
    def __init__(self,MM_max,MM_min):
        self.max=MM_max
        self.min=MM_min

def lzq_load_data(len_test,len_closeness,len_period,T_closeness=1,T_period=24*2):
    
    all_data=np.load('DATA/data_XY3_T/DATA_XY3_T1.npy')
    #import matplotlib.pyplot as plt
    #ave=np.mean(np.mean(np.mean(all_data,axis=1),axis=1),axis=1)
    #plt.plot(ave)
    len_total,feature,map_height,map_width=all_data.shape
    #all_data=np.arange(2880*2*1*1).reshape(-1,2,1,1)
    #len_total,feature,map_height,map_width=all_data.shape
    print('all_data shape: ',all_data.shape)
    mm=MM(np.max(all_data[:-len_test]),np.min(all_data[:-len_test]))
    print('max=',mm.max,' min=',mm.min)
    
    #for time
    time=np.arange(len_total,dtype=int)
    #hour
    time_hour=time%T_period
    matrix_hour=np.zeros([len_total,T_period,map_height,map_width])
    for i in range(len_total):
        matrix_hour[i,time_hour[i],:,:]=1
    #day
    time_day=(time//T_period)%7
    matrix_day=np.zeros([len_total,7,map_height,map_width])
    for i in range(len_total):
        matrix_day[i,time_day[i],:,:]=1
    #con
    matrix_T=np.concatenate((matrix_hour,matrix_day),axis=1)
    
    all_data=(2.0*all_data-(mm.max+mm.min))/(mm.max-mm.min)
    print('mean=',np.mean(all_data),' variance=',np.std(all_data))
    
    if len_period>0:
        number_of_skip_hours=T_period*len_period
    elif len_closeness>0:
        number_of_skip_hours=T_closeness*len_closeness  
    else:
        print("wrong")
    print('number_of_skip_hours:',number_of_skip_hours)
    
    Y=all_data[number_of_skip_hours:len_total]

    if len_closeness>0:
        X_closeness=all_data[number_of_skip_hours-T_closeness*1:len_total-T_closeness*1]
        for i in range(len_closeness-1):
            X_closeness=np.concatenate((X_closeness,all_data[number_of_skip_hours-T_closeness*(2+i):len_total-T_closeness*(2+i)]),axis=1)
    if len_period>0:
        X_period=all_data[number_of_skip_hours-T_period*1:len_total-T_period*1]
        for i in range(len_period-1):
            X_period=np.concatenate((X_period,all_data[number_of_skip_hours-T_period*(2+i):len_total-T_period*(2+i)]),axis=1)
   
    
    matrix_T=matrix_T[number_of_skip_hours:]
    
    X_closeness_train=X_closeness[:-len_test] 
    X_period_train=X_period[:-len_test] 
     
    T_train=matrix_T[:-len_test] 
    
    X_closeness_test=X_closeness[-len_test:] 
    X_period_test=X_period[-len_test:] 
              
    T_test=matrix_T[-len_test:]         
    
    
    X_train=[X_closeness_train,X_period_train]
    X_test=[X_closeness_test,X_period_test]
    #X_train=np.concatenate((X_closeness_train,X_period_train,X_trend_train),axis=1)
    #X_test=np.concatenate((X_closeness_test,X_period_test,X_trend_test),axis=1)
    Y_train=Y[:-len_test] 
    Y_test=Y[-len_test:] 

    len_train=X_closeness_train.shape[0]
    len_test=X_closeness_test.shape[0]
    
    poi=np.load('DATA/data_XY3_T/POI_XY3.npy')
    
    for i in range(poi.shape[0]):
        poi[i]=poi[i]/np.max(poi[i])
        
    P_train=np.repeat(poi.reshape(1,poi.shape[0],map_height,map_width),len_train,axis=0)
    P_test =np.repeat(poi.reshape(1,poi.shape[0],map_height,map_width),len_test ,axis=0)
    
    return X_train,T_train,P_train,Y_train,X_test,T_test,P_test,Y_test,mm.max-mm.min


