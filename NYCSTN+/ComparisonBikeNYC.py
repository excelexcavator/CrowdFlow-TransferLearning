#from __future__ import print_function
from DATA.lzq_read_data_time_poi import lzq_load_data
from keras.callbacks import ModelCheckpoint
#import cPickle as pickle
import numpy as np
#import math

NO=4
#for reproduction
seed=1
for i in range(NO):
    seed=seed*10+7
seed=seed*10+7
np.random.seed(seed)
#from ipdb import set_trace
#set_trace()

#for GPU in Lab
device=6

import os
os.environ["CUDA_VISIBLE_DEVICES"]=str(device)
import tensorflow as tf  #from V1707
config=tf.ConfigProto()  #from V1707
config.gpu_options.allow_growth=True  #from V1707
#config.gpu_options.per_process_gpu_memory_fraction=0.5
sess=tf.Session(config=config)  #from V1707
#import keras.backend.tensorflow_backend as KTF
#KTF._set_session(tf.Session(config=config))
import setproctitle  #from V1707
setproctitle.setproctitle('Comprison Start! @ ZiqianLin')  #from V1707

from keras import backend as K
K.set_image_data_format('channels_first')


#hyperparameters
epoch = 350  # number of epoch at training stage
batch_size = 32  # batch size
lr = 0.0002  # learning rate

H,W,channel = 21,12,2   # grid size

T = 24*1  # number of time intervals in one day

len_closeness = 3  # length of closeness dependent sequence
len_period = 4  # length of peroid dependent sequence
len_trend = 4  # length of trend dependent sequence

T_closeness,T_period,T_trend=1,T,T*7

# last 7 days for testing data
days_test = 14
len_test = T * days_test

#the number of repetition and if retrain the model
iterate_num=10
trainDST=0  #DST
train11=0   #DSTN+ResPlus+PoI&Time
train10=0   #DSTN+ResPlus
train01=0   #DSTN+PoI&Time
train00=0   #DSTN



#DST result
setproctitle.setproctitle('BJMobile DST @ ZiqianLin')  #from V1707

print("loading data...")
X_train,T_train,P_train,Y_train,X_test,T_test,P_test,Y_test,MM=lzq_load_data(len_test,len_closeness,len_period,len_trend,T_closeness,T_period,T_trend)

R_N = 4   # number of residual units

from keras.optimizers import Adam
from DST_network.STResNet import stresnet
import DST_network.metrics as metrics

def build_model(external_dim,CFN):
    c_conf = (len_closeness, channel, H, W) if len_closeness > 0 else None
    p_conf = (len_period,    channel, H, W) if len_period    > 0 else None
    t_conf = (len_trend,     channel, H, W) if len_trend     > 0 else None

    model = stresnet(c_conf=c_conf, p_conf=p_conf, t_conf=t_conf,
                     external_dim=external_dim, nb_residual_unit=R_N, CF=CFN)
    
    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse,metrics.mae])
    model.summary()
    #from keras.utils.visualize_util import plot
    #plot(model, to_file='model.png', show_shapes=True)
    return model


CF=64

iterate_loop=np.arange(iterate_num)+1+iterate_num*(NO-1)

RMSE=np.zeros([iterate_num,1])
MAE =np.zeros([iterate_num,1])
count_sum=iterate_num

import time

count=0

    
for iterate_index in range(iterate_num):
    count=count+1
    iterate=iterate_loop[iterate_index]
        
    time_start=time.time()
        
    F='DST_MODEL/dst_model_'+str(iterate)+'_.hdf5'
        
    model = build_model(external_dim=False,CFN=CF)
    if trainDST:
        model_checkpoint=ModelCheckpoint(
            filepath=F,
            monitor='val_rmse',
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            period=1)
            
        print('=' * 10)
        print("training model...")
        history = model.fit(X_train, Y_train,
                            nb_epoch=epoch,
                            batch_size=batch_size,
                            validation_split=0.1,
                            callbacks=[model_checkpoint],
                            verbose=1)
        
    print('=' * 10)
    print('evaluating using the model that has the best loss on the valid set')
    model.load_weights(F)
    
    score = model.evaluate(X_train, Y_train, batch_size=Y_train.shape[0] // 48, verbose=0)
    print('              mse     rmse    mae')
    print('Train score:',end=' ')
    np.set_printoptions(precision=6, suppress=True)
    score = model.evaluate(X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)
    print('Test  score:',end=' ')
    np.set_printoptions(precision=6, suppress=True)
    print(np.array(score))
        
    RMSE[iterate_index,0]=score[1]
    MAE [iterate_index,0]=score[2]
        
    for_show=np.concatenate([RMSE,MAE],axis=1)*MM/2

    np.set_printoptions(precision=4, suppress=True)
    print('RMSE  MAE')
    print(for_show)
       
    for_show=np.mean(for_show,axis=0)
    print('RMSE  MAE')
    print(for_show)
    
    np.save('DST_SCORE/dst_score.npy',[RMSE,MAE])
        
    time_end=time.time()
        
    print('totally cost',time_end-time_start)
    print(str(count)+'/'+str(count_sum))



#DSTN+ResPlus+PoI&Time
setproctitle.setproctitle('BJMobile DSTN+ResPlus+PoI&Time @ ZiqianLin')  #from V1707
from PPT3_network.PPT3net import PPT3

X_train,T_train,P_train,Y_train,X_test,T_test,P_test,Y_test,MM=lzq_load_data(len_test,len_closeness,len_period,len_trend,T_closeness,T_period,T_trend)
   
X_train=np.concatenate((X_train[0],X_train[1],X_train[2]),axis=1)
X_test =np.concatenate((X_test[0], X_test[1], X_test[2] ),axis=1)

index=np.arange(9)
P_train=P_train[:,index,:,:]
P_test =P_test [:,index,:,:]

pre_F=64
conv_F=64
R_N=2
   
is_plus=True
plus=8
rate=1
   
is_pt=True
P_N=9
T_F=7*8
PT_F=9

drop=0.1

import time
count=0
count_sum=iterate_num

iterate_loop=np.arange(iterate_num)+1+iterate_num*(NO-1)

RMSE=np.zeros([iterate_num,1])
MAE =np.zeros([iterate_num,1])

for iterate_index in range(iterate_num):
    count=count+1
    time_start=time.time()       
    iterate=iterate_loop[iterate_index]
    
    print("***** conv_model *****")
    model=PPT3(H=H,W=W,channel=channel,
                       c=len_closeness,p=len_period,                
                       pre_F=pre_F,conv_F=conv_F,R_N=R_N,    
                       is_plus=is_plus,
                       plus=plus,rate=rate,     
                       is_pt=is_pt,P_N=P_N,T_F=T_F,PT_F=PT_F,T=T,     
                       drop=drop)            
    
    file_conv='PPT3_11/MODEL/PPT3_11_model_'+str(iterate)+'.hdf5'
    #train conv_model
    if train11:
        model_checkpoint=ModelCheckpoint(
                filepath=file_conv,
                monitor='val_rmse',
                verbose=1,
                save_best_only=True,
                save_weights_only=True,
                mode='min',
                period=1
            )
        
        print('=' * 10)
        print("***** training conv_model *****")
        history = model.fit([X_train,P_train,T_train], Y_train,
                            epochs=epoch,
                            batch_size=batch_size,
                            validation_split=0.1,
                            callbacks=[model_checkpoint],
                            verbose=1)
        
    print('=' * 10)
    print('***** evaluate *****')
    model.load_weights(file_conv)
    
    score = model.evaluate([X_train,P_train,T_train], Y_train, batch_size=Y_train.shape[0] // 48, verbose=0)
    print('              mse     rmse    mae')
    print('Train score:',end=' ')
    np.set_printoptions(precision=6, suppress=True)
    print(np.array(score))
    score = model.evaluate([X_test ,P_test ,T_test ], Y_test, batch_size=Y_test.shape[0], verbose=0)
    print('Test  score:',end=' ')
    np.set_printoptions(precision=6, suppress=True)
    print(np.array(score))
    
    RMSE[iterate_index,0]=score[1]
    MAE [iterate_index,0]=score[2]
    
    for_show=np.concatenate([RMSE,MAE],axis=1)*MM/2

    np.set_printoptions(precision=4, suppress=True)
    print('RMSE  MAE')
    print(for_show)
       
    for_show=np.mean(for_show,axis=0)
    print('RMSE  MAE')
    print(for_show)
    
    np.save('PPT3_11/SCORE/PPT3_11_score3.npy',[RMSE,MAE])
        
    time_end=time.time()
    print('iterate cost',time_end-time_start)
    print(str(count)+'/'+str(count_sum))



#DSTN+ResPlus
setproctitle.setproctitle('BJMobile DSTN+ResPlus @ ZiqianLin')  #from V1707
from PPT3_network.PPT3net import PPT3

X_train,T_train,P_train,Y_train,X_test,T_test,P_test,Y_test,MM=lzq_load_data(len_test,len_closeness,len_period,len_trend,T_closeness,T_period,T_trend)
   
X_train=np.concatenate((X_train[0],X_train[1],X_train[2]),axis=1)
X_test =np.concatenate((X_test[0], X_test[1], X_test[2] ),axis=1)

#index=np.arange(9)
#P_train=P_train[:,index,:,:]
#P_test =P_test [:,index,:,:]

pre_F=64
conv_F=64
R_N=2
   
is_plus=True
plus=8
rate=1
   
is_pt=False
P_N=9
T_F=7*8
PT_F=9

drop=0.1

import time
count=0
count_sum=iterate_num

iterate_loop=np.arange(iterate_num)+1+iterate_num*(NO-1)

RMSE=np.zeros([iterate_num,1])
MAE =np.zeros([iterate_num,1])

for iterate_index in range(iterate_num):
    count=count+1
    time_start=time.time()       
    iterate=iterate_loop[iterate_index]
    
    print("***** conv_model *****")
    model=PPT3(H=H,W=W,channel=channel,
                       c=len_closeness,p=len_period,                
                       pre_F=pre_F,conv_F=conv_F,R_N=R_N,    
                       is_plus=is_plus,
                       plus=plus,rate=rate,     
                       is_pt=is_pt,P_N=P_N,T_F=T_F,PT_F=PT_F,T=T,     
                       drop=drop)            
    
    file_conv='PPT3_10/MODEL/PPT3_10_model_'+str(iterate)+'.hdf5'
    #train conv_model
    if train10:
        model_checkpoint=ModelCheckpoint(
                filepath=file_conv,
                monitor='val_rmse',
                verbose=1,
                save_best_only=True,
                save_weights_only=True,
                mode='min',
                period=1
            )
        
        print('=' * 10)
        print("***** training conv_model *****")
        history = model.fit(X_train, Y_train,
                            epochs=epoch,
                            batch_size=batch_size,
                            validation_split=0.1,
                            callbacks=[model_checkpoint],
                            verbose=1)
        
    print('=' * 10)
    print('***** evaluate *****')
    model.load_weights(file_conv)
    
    score = model.evaluate(X_train, Y_train, batch_size=Y_train.shape[0] // 48, verbose=0)
    print('              mse     rmse    mae')
    print('Train score:',end=' ')
    np.set_printoptions(precision=6, suppress=True)
    print(np.array(score))
    score = model.evaluate(X_test,  Y_test, batch_size=Y_test.shape[0], verbose=0)
    print('Test  score:',end=' ')
    np.set_printoptions(precision=6, suppress=True)
    print(np.array(score))
    
    RMSE[iterate_index,0]=score[1]
    MAE [iterate_index,0]=score[2]
    
    for_show=np.concatenate([RMSE,MAE],axis=1)*MM/2

    np.set_printoptions(precision=4, suppress=True)
    print('RMSE  MAE')
    print(for_show)
       
    for_show=np.mean(for_show,axis=0)
    print('RMSE  MAE')
    print(for_show)
    
    np.save('PPT3_10/SCORE/PPT3_10_score.npy',[RMSE,MAE])
        
    time_end=time.time()
    print('iterate cost',time_end-time_start)
    print(str(count)+'/'+str(count_sum))
    
    
    
#DSTN+PoI&Time
setproctitle.setproctitle('BJMobile DSTN+PoI&Time @ ZiqianLin')  #from V1707
from PPT3_network.PPT3net import PPT3

X_train,T_train,P_train,Y_train,X_test,T_test,P_test,Y_test,MM=lzq_load_data(len_test,len_closeness,len_period,len_trend,T_closeness,T_period,T_trend)
  
X_train=np.concatenate((X_train[0],X_train[1],X_train[2]),axis=1)
X_test =np.concatenate((X_test[0], X_test[1], X_test[2] ),axis=1)

#index=np.arange(9)
#P_train=P_train[:,index,:,:]
#P_test =P_test [:,index,:,:]

pre_F=64
conv_F=64
R_N=2
   
is_plus=False
plus=8
rate=1
   
is_pt=True
P_N=9
T_F=7*8
PT_F=9

drop=0.1

import time
count=0
count_sum=iterate_num

iterate_loop=np.arange(iterate_num)+1+iterate_num*(NO-1)

RMSE=np.zeros([iterate_num,1])
MAE =np.zeros([iterate_num,1])

for iterate_index in range(iterate_num):
    count=count+1
    time_start=time.time()       
    iterate=iterate_loop[iterate_index]
    
    print("***** conv_model *****")
    model=PPT3(H=H,W=W,channel=channel,
                       c=len_closeness,p=len_period,                
                       pre_F=pre_F,conv_F=conv_F,R_N=R_N,    
                       is_plus=is_plus,
                       plus=plus,rate=rate,     
                       is_pt=is_pt,P_N=P_N,T_F=T_F,PT_F=PT_F,T=T,     
                       drop=drop)            
    
    file_conv='PPT3_01/MODEL/PPT3_01_model_'+str(iterate)+'.hdf5'
    #train conv_model
    if train01:
        model_checkpoint=ModelCheckpoint(
                filepath=file_conv,
                monitor='val_rmse',
                verbose=1,
                save_best_only=True,
                save_weights_only=True,
                mode='min',
                period=1
            )
        
        print('=' * 10)
        print("***** training conv_model *****")
        history = model.fit([X_train,P_train,T_train], Y_train,
                            epochs=epoch,
                            batch_size=batch_size,
                            validation_split=0.1,
                            callbacks=[model_checkpoint],
                            verbose=1)
        
    print('=' * 10)
    print('***** evaluate *****')
    model.load_weights(file_conv)
    
    score = model.evaluate([X_train,P_train,T_train], Y_train, batch_size=Y_train.shape[0] // 48, verbose=0)
    print('              mse     rmse    mae')
    print('Train score:',end=' ')
    np.set_printoptions(precision=6, suppress=True)
    print(np.array(score))
    score = model.evaluate([X_test, P_test, T_test ],  Y_test,  batch_size=Y_test.shape[0], verbose=0)
    print('Test  score:',end=' ')
    np.set_printoptions(precision=6, suppress=True)
    print(np.array(score))
    
    RMSE[iterate_index,0]=score[1]
    MAE [iterate_index,0]=score[2]
    
    for_show=np.concatenate([RMSE,MAE],axis=1)*MM/2

    np.set_printoptions(precision=4, suppress=True)
    print('RMSE  MAE')
    print(for_show)
       
    for_show=np.mean(for_show,axis=0)
    print('RMSE  MAE')
    print(for_show)
    
    np.save('PPT3_01/SCORE/PPT3_01_score.npy',[RMSE,MAE])
        
    time_end=time.time()
    print('iterate cost',time_end-time_start)
    print(str(count)+'/'+str(count_sum))
 
    

#DSTN
setproctitle.setproctitle('BJMobile DSTN+PoI&Time @ ZiqianLin')  #from V1707
from PPT3_network.PPT3net import PPT3

X_train,T_train,P_train,Y_train,X_test,T_test,P_test,Y_test,MM=lzq_load_data(len_test,len_closeness,len_period,len_trend,T_closeness,T_period,T_trend)
   
X_train=np.concatenate((X_train[0],X_train[1],X_train[2]),axis=1)
X_test =np.concatenate((X_test[0], X_test[1], X_test[2] ),axis=1)

index=np.arange(9)
P_train=P_train[:,index,:,:]
P_test =P_test [:,index,:,:]

pre_F=64
conv_F=64
R_N=2
   
is_plus=False
plus=8
rate=1
   
is_pt=False
P_N=9
T_F=7*8
PT_F=9

drop=0.1

import time
count=0
count_sum=iterate_num

iterate_loop=np.arange(iterate_num)+1+iterate_num*(NO-1)

RMSE=np.zeros([iterate_num,1])
MAE =np.zeros([iterate_num,1])

for iterate_index in range(iterate_num):
    count=count+1
    time_start=time.time()       
    iterate=iterate_loop[iterate_index]
    
    print("***** conv_model *****")
    model=PPT3(H=H,W=W,channel=channel,
                       c=len_closeness,p=len_period,                
                       pre_F=pre_F,conv_F=conv_F,R_N=R_N,    
                       is_plus=is_plus,
                       plus=plus,rate=rate,     
                       is_pt=is_pt,P_N=P_N,T_F=T_F,PT_F=PT_F,T=T,     
                       drop=drop)            
    
    file_conv='PPT3_00/MODEL/PPT3_00_model_'+str(iterate)+'.hdf5'
    #train conv_model
    if train00:
        model_checkpoint=ModelCheckpoint(
                filepath=file_conv,
                monitor='val_rmse',
                verbose=1,
                save_best_only=True,
                save_weights_only=True,
                mode='min',
                period=1
            )
        
        print('=' * 10)
        print("***** training conv_model *****")
        history = model.fit(X_train, Y_train,
                            epochs=epoch,
                            batch_size=batch_size,
                            validation_split=0.1,
                            callbacks=[model_checkpoint],
                            verbose=1)
        
    print('=' * 10)
    print('***** evaluate *****')
    model.load_weights(file_conv)
    
    score = model.evaluate(X_train, Y_train, batch_size=Y_train.shape[0] // 48, verbose=0)
    print('              mse     rmse    mae')
    print('Train score:',end=' ')
    np.set_printoptions(precision=6, suppress=True)
    print(np.array(score))
    score = model.evaluate(X_test,  Y_test,  batch_size=Y_test.shape[0], verbose=0)
    print('Test  score:',end=' ')
    np.set_printoptions(precision=6, suppress=True)
    print(np.array(score))
    
    RMSE[iterate_index,0]=score[1]
    MAE [iterate_index,0]=score[2]
    
    for_show=np.concatenate([RMSE,MAE],axis=1)*MM/2

    np.set_printoptions(precision=4, suppress=True)
    print('RMSE  MAE')
    print(for_show)
       
    for_show=np.mean(for_show,axis=0)
    print('RMSE  MAE')
    print(for_show)
    
    np.save('PPT3_00/SCORE/PPT3_00_score.npy',[RMSE,MAE])
        
    time_end=time.time()
    print('iterate cost',time_end-time_start)
    print(str(count)+'/'+str(count_sum))



#Comparison
X_train,T_train,P_train,Y_train,X_test,T_test,P_test,Y_test,MM=lzq_load_data(len_test,len_closeness,len_period,len_trend,T_closeness,T_period,T_trend)
 
np.set_printoptions(precision=4, suppress=True)
print('MODEL                     RMSE  MAE')

print('ResNet                  :',end=' ')
[RMSE,MAE]=np.load('DST_SCORE/dst_score.npy')
for_show=np.concatenate([RMSE,MAE],axis=1)*MM/2
print(for_show)   

print('DeepSTN                 :',end=' ')
[RMSE,MAE]=np.load('PPT3_00/SCORE/PPT3_00_score.npy')
for_show=np.concatenate([RMSE,MAE],axis=1)*MM/2
for_show=np.mean(for_show,axis=0)
print(for_show)   

print('DeepSTN+ResPlus         :',end=' ')
[RMSE,MAE]=np.load('PPT3_10/SCORE/PPT3_10_score.npy')
for_show=np.concatenate([RMSE,MAE],axis=1)*MM/2
for_show=np.mean(for_show,axis=0)
print(for_show)   

print('DeepSTN+PoI&Time        :',end=' ')
[RMSE,MAE]=np.load('PPT3_01/SCORE/PPT3_01_score.npy')
for_show=np.concatenate([RMSE,MAE],axis=1)*MM/2
for_show=np.mean(for_show,axis=0)
print(for_show)   

print('DeepSTN+ResPlus+PoI&Time:',end=' ')
[RMSE,MAE]=np.load('PPT3_11/SCORE/PPT3_11_score3.npy')
for_show=np.concatenate([RMSE,MAE],axis=1)*MM/2
for_show=np.mean(for_show,axis=0)
print(for_show)   
