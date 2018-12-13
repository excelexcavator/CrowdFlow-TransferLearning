from __future__ import absolute_import, division, print_function
import scipy.io as scio
import tensorflow as tf
import pandas as pd
import numpy as np
import _pickle as cPickle
import keras
import json



#   lat, lon, time
map_beijing = np.zeros([50, 50, 1440])
with open("population_tencent_beijing_2000w.json", 'r') as file_beijing:
    load_dict = json.load(file_beijing)
loc = load_dict['centers']['count']
pop = load_dict['population']['count']

for i in range(len(loc)):
    lon = loc[i][0]
    lat = loc[i][1]
    lon = int(lon * 100) - 11655
    lat = int(lat * 100) - 4000
    if 0 <= lon < 50 and 0 <= lat < 50:
        map_beijing[lat][lon] += pop[i]

cPickle.dump(map_beijing, open('map_beijing_small.pkl', 'wb'), protocol=2)
# usage : map_beijing = cPickle.load(open('map_beijing.pkl', 'rb'))

map_shanghai = np.zeros([50, 50, 4464])
file_shanghai = 'shanghai.mat'
load_mat = scio.loadmat(file_shanghai)
loc = load_mat['data_loc']
pop = load_mat['people']
for i in range(len(loc)):
    lon = list(loc[i])[0]
    lat = list(loc[i])[1]
    lon = int(lon * 100) - 12121
    lat = int(lat * 100) - 3110
    if 0 <= lon < 50 and 0 <= lat < 50:
        map_shanghai[lat][lon] += list(pop[i])

cPickle.dump(map_shanghai, open('map_shanghai_small.pkl', 'wb'), protocol=2)