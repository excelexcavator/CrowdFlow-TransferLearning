from __future__ import absolute_import, division, print_function
from sklearn.model_selection import train_test_split
from tools import ShowProcess
import time
import tensorflow as tf
import pandas as pd
import numpy as np
import _pickle as cPickle
import keras
import keras.backend.tensorflow_backend as KTF
import setproctitle

setproctitle.setproctitle('crowdflow_transfer@changshuhao')
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
KTF.set_session(tf.Session(config=config))

map_beijing = cPickle.load(open('map_beijing.pkl', 'rb'))    # 220 * 170 * 1440
map_shanghai = cPickle.load(open('map_shanghai.pkl', 'rb'))  # 122 * 136 * 4464

lon_beijing, lat_beijing, time_beijing = map_beijing.shape
lon_shanghai, lat_shanghai, time_shanghai = map_shanghai.shape
lon_max = max([lon_beijing, lon_shanghai])
lat_max = max([lat_beijing, lat_shanghai])
lon_pad_beijing = int((lon_max - lon_beijing) / 2)
lat_pad_beijing = int((lat_max - lat_beijing) / 2)
lon_pad_shanghai = int((lon_max - lon_shanghai) / 2)
lat_pad_shanghai = int((lat_max - lat_shanghai) / 2)
period = 1 * 24   # 1 day
block_num_beijing = int((map_beijing.shape[2] - period) / 4)    # number of training blocks /2
block_num_shanghai = int((map_shanghai.shape[2] - period) / 12)      # number of training blocks
data = np.zeros([block_num_beijing + block_num_shanghai, lon_max, lat_max, period])
predict_label = np.zeros([block_num_beijing + block_num_shanghai, lon_max, lat_max, 1])
domain_label = np.zeros([block_num_beijing + block_num_shanghai, 1])

for i in range(block_num_beijing):
    temp = map_beijing[:, :, i:i+25]
    temp = np.pad(temp, ((lon_pad_beijing, lon_pad_beijing),(lat_pad_beijing, lat_pad_beijing),(0,0)), 'constant')
    data[i] = temp[:, :, 0:24]
    predict_label[i] = temp[:, :, 24].reshape(lon_max, lat_max, 1)
    domain_label[i] = 0
for i in range(block_num_shanghai):
    temp = map_shanghai[:, :, i:i+25]
    temp = np.pad(temp, ((lon_pad_shanghai, lon_pad_shanghai),(lat_pad_shanghai, lat_pad_shanghai),(0,0)), 'constant')
    data[i + block_num_beijing] = temp[:, :, 0:24]
    predict_label[i + block_num_beijing] = temp[:, :, 24].reshape(lon_max, lat_max, 1)
    domain_label[i + block_num_beijing] = 1

print('========== finished loading ==========')

train_data, test_data, train_predict_label, test_predict_label, train_domain_label, test_domain_label = train_test_split(
    data, predict_label, domain_label, test_size=0.2, random_state=int(time.time()) % 49999)
train_num = train_data.shape[0]
test_num = test_data.shape[0]
print('train_data',train_data.shape)
print('test_data',test_data.shape)
print('train_predict_label',train_predict_label.shape)
print('test_predict_label',test_predict_label.shape)
print('train_domain_label',train_domain_label.shape)
print('test_domain_label',test_domain_label.shape)


class TransferNet:
    def build_FeatureNet(inputs):
        x = keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu, padding="same")(inputs)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu, padding="same")(x)
        return x

    def build_PredictNet(inputs):
        x = keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu, padding="same")(inputs)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu, padding="same")(x)
        x = keras.layers.Dense(1, name="predict_output")(x)
        return x

    def build_DomainNet(inputs):
        x = keras.layers.Dense(16, activation=tf.nn.relu)(inputs)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(1, activation=tf.nn.sigmoid, name="domain_output")(x)
        return x

    def build(lon, lat, period):
        inputs = keras.layers.Input(shape=(lon, lat, period))
        region_representation = TransferNet.build_FeatureNet(inputs)
        predict_branch = TransferNet.build_PredictNet(region_representation)
        domain_branch = TransferNet.build_DomainNet(region_representation)
        model = keras.Model(inputs=inputs, outputs=[predict_branch, domain_branch])
        return model


model = TransferNet.build(lon_max, lat_max, period)
losses = {
    "predict_output": "mse",
    "domain_output": "binary_crossentropy"
}
loss_weights = {
    "predict_output": 1.0,
    "domain_output": -0.1
}
optimizer = tf.train.AdamOptimizer(0.001)
model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=["accuracy"])
model.summary()

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        process_bar.show_process()

EPOCHS = 200
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
process_bar = ShowProcess(EPOCHS, '')
history = model.fit(train_data,
                    {"predict_output": train_predict_label.reshape([train_num, lon_max, lat_max, 1]),
                     "domain_output": train_domain_label},
                    epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[early_stop, PrintDot()])