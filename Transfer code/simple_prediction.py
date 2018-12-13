from __future__ import absolute_import, division, print_function
from tools import ShowProcess
import tensorflow as tf
import numpy as np
import cPickle
import keras
from sklearn.model_selection import train_test_split
import keras.backend.tensorflow_backend as KTF
import setproctitle

setproctitle.setproctitle('crowdflow_transfer@changshuhao')
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
KTF.set_session(tf.Session(config=config))

map_beijing = cPickle.load(open('map_beijing.pkl', 'rb'))   # 220 * 170 * 1440
#map_shanghai = cPickle.load(open('map_shanghai.pkl', 'rb'))

period = 1 * 24   # 1 day
block_num = int(map_beijing.shape[2] - period)    # number of training blocks
data = []
label = []
for i in range(block_num):
    data.append(map_beijing[:, :, i:i+24])
    label.append(map_beijing[:, :, i+24])

data = np.array(data)
label = np.array(label)
train_data, test_data, train_label, test_label = train_test_split(
    data, label, test_size=0.33, random_state=7)
train_num = train_data.shape[0]
test_num = test_data.shape[0]
print('train_data',train_data.shape)
print('train_label',train_label.shape)
print('test_data',test_data.shape)
print('test_label',test_label.shape)

def build_model():
  model = keras.models.Sequential([
      keras.layers.Conv2D(32, (1,1), activation=tf.nn.relu, input_shape=train_data.shape[1:]),
      keras.layers.Conv2D(32, (3,3), activation=tf.nn.relu, padding="same"),
      keras.layers.Dropout(0.2),
      keras.layers.Conv2D(32, (1,1), activation=tf.nn.relu, padding="same"),
      keras.layers.Conv2D(32, (3,3), activation=tf.nn.relu, padding="same"),
      keras.layers.Dropout(0.2),
      keras.layers.Conv2D(16, (1,1), activation=tf.nn.relu, padding="same"),
      keras.layers.Conv2D(16, (3,3), activation=tf.nn.relu, padding="same"),
      keras.layers.Conv2D(1, (1,1))
  ])

  optimizer = tf.train.RMSPropOptimizer(0.0001)
  model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
  return model

model = build_model()
model.summary()

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        process_bar.show_process()

EPOCHS = 200
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
process_bar = ShowProcess(EPOCHS, '')
history = model.fit(train_data, train_label.reshape([train_num, 220, 170, 1]), epochs=EPOCHS,
                    validation_split=0.2, verbose=0, batch_size=24,
                    callbacks=[early_stop, PrintDot()])


train_loss = history.history['mean_absolute_error']
val_loss = history.history['val_mean_absolute_error']
file_tl = open('train_loss.txt','a')
file_vl = open('val_loss.txt','a')
file_tl.write(str(train_loss))
file_vl.write(str(val_loss))
file_vl.close()
file_tl.close()
[loss, mae] = model.evaluate(test_data, test_label.reshape([test_num, 220, 170, 1]), verbose=0)
print("Testing set Mean Abs Error: {:1.3f}".format(mae))
