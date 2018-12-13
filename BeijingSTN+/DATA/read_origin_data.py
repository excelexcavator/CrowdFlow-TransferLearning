import json

if 0:
    #try json
    save_dict = {'bigberg': [7600, {1: [['iPhone', 6300], ['Bike', 800], ['shirt', 300]]}]}
    
    with open("try.json","w") as f:
        json.dump(save_dict,f)
        print("加载入文件完成...")
    
    #load_dict = json.loads('try.json') faied
    with open("try.json",'r') as load_f:
        load_dict = json.load(load_f)

if 1:
    file_name = "newdata/population_tencent_beijing_2000w_15min_3000m.json"
    with open(file_name,'r') as load_f:
        data_dict = json.load(load_f)
        print("load data finished")

print(data_dict.keys())

population = data_dict['population']
centers = data_dict['centers']

print(population.keys())
print(centers.keys())

#pop_ct = population['count']
pop_ou = population['out']
pop_in = population['in']

#cen_ct = centers['count']
cen_ou = centers['out']
cen_in = centers['in']


import numpy as np

def list_switch(L):
    L=np.array(L)
    L=np.transpose(L,(1,0))
    #L=L.tolist()
    return L

#cen_ct=list_switch(cen_ct)
cen_ou=list_switch(cen_ou)
cen_in=list_switch(cen_in)


print(np.max(cen_in[0,:])==np.max(cen_ou[0,:]))#,np.max(cen_in[0,:])==np.max(cen_ct[0,:]))
print(np.min(cen_in[0,:])==np.min(cen_ou[0,:]))#,np.min(cen_in[0,:])==np.min(cen_ct[0,:]))
print(np.max(cen_in[1,:])==np.max(cen_ou[1,:]))#,np.max(cen_in[1,:])==np.max(cen_ct[1,:]))
print(np.min(cen_in[1,:])==np.min(cen_ou[1,:]))#,np.min(cen_in[1,:])==np.min(cen_ct[1,:]))


X_max=np.max(cen_in[0,:])
X_min=np.min(cen_in[0,:])
Y_max=np.max(cen_in[1,:])
Y_min=np.min(cen_in[1,:])

H=22#66
W=24#73

np.save('newdata/XYmaxmin3000.npy',[X_max,X_min,H,Y_max,Y_min,W])

def grid(array):
    array[0,:]=(array[0,:]-X_min)*(H-1)/(X_max-X_min)
    array[1,:]=(array[1,:]-Y_min)*(W-1)/(Y_max-Y_min)
    return array

#cen_ct=grid(cen_ct)
cen_ou=grid(cen_ou)
cen_in=grid(cen_in)



import matplotlib.pyplot as plt

cut1=0
cut2=H*W

#print('cen_ct')
#plt.scatter(cen_ct[0,cut1:cut2],cen_ct[1,cut1:cut2],c='red',marker='.')
#plt.axis([0, 10, 0, 10])
#plt.show()

print('cen_ou')
plt.scatter(cen_ou[0,cut1:cut2],cen_ou[1,cut1:cut2],c='red',marker='.')
plt.axis([-1, H+1, -1, W+1])
plt.show()

print('cen_in')
plt.scatter(cen_in[0,cut1:cut2],cen_in[1,cut1:cut2],c='red',marker='.')
plt.axis([-1, H+1, -1, W+1])
plt.show()

#a=(cen_ct==cen_in)*(cen_ct==cen_in)
#print(sum(a[0,:]*a[1,:]))


#cen_ct = np.round(cen_ct)
cen_ou = np.round(cen_ou)
cen_in = np.round(cen_in)

#cen_ct = cen_ct.astype(int)
cen_ou = cen_ou.astype(int)
cen_in = cen_in.astype(int)


data = np.zeros([2880,2,H,W],float)

for block in range(H*W):
    print(block)
    data[:,0,cen_in[0,block],cen_in[1,block]] = pop_in[block]
    data[:,1,cen_ou[0,block],cen_ou[1,block]] = pop_ou[block]

data0=data[:,0,:,:]
data1=data[:,1,:,:]

np.save('newdata/NewData3000.npy',data)

