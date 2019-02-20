from PIL import Image
from tqdm import tqdm
import os
import tensorflow as tf
import numpy as np
import dataset

def next_batch(num, data, labels):

    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def compute_loss(v_xs, v_ys):
    global logits
    y_pre = sess.run(logits, feed_dict={xs: v_xs})
#     loss = tf.losses.absolute_difference(labels=v_ys, predictions=y_pre)
    loss = tf.abs(v_ys-y_pre)
    result = sess.run(loss, feed_dict={xs: v_xs, ys: v_ys})
    return result


# train_data = dataset.CCPD5000_data('./ccpd5000/train/')
# train_labels = dataset.CCPD5000_labels('./ccpd5000/train/')

# print(len(train_data))
# print(len(train_labels))
# img = train_data[-1]
# kpt = train_labels[-1]
# print(type(img))
# print(type(kpt))
# # img = tf.cast(img, tf.float32)



# data_batch, label_batch = next_batch(5, train_data, train_labels)
# print('\n5 random samples')
# print(type(data_batch))

# data_batch.shape


xs = tf.placeholder(tf.float32, [None,320,192,3],name="xs") 
ys = tf.placeholder(tf.float32, [None, 8],name="ys")
# Input Layer
input_layer = tf.reshape(xs, [-1, 320, 192, 3])

  # Convolutional Layer #1 [-1,320,192,32]
conv1 = tf.layers.conv2d(inputs=input_layer,filters=32,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
  # Pooling Layer #1 [-1,160,96,32]
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
  # Convolutional Layer #2  [-1,160,96,32]
conv2 = tf.layers.conv2d(inputs=pool1,filters=32,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
  #  Pooling Layer #2 [-1,80,48,32]
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  # Convolutional Layer #3  [-1,80,48,64]
conv3 = tf.layers.conv2d(inputs=pool2,filters=64,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
  # Pooling Layer #3 [-1,40,24,64]
pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
  # Convolutional Layer #4  [-1,40,24,64]
conv4 = tf.layers.conv2d(inputs=pool3,filters=64,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
  #  Pooling Layer #4 [-1,20,12,64]
pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

  # Dense Layer
pool4_flat = tf.reshape(pool4, [-1, 20 * 12 * 64])
dense1 = tf.layers.dense(inputs=pool4_flat, units=128, activation=tf.nn.leaky_relu)
  # Logits Layer
logits = tf.layers.dense(inputs=dense1, units=8,activation=tf.nn.sigmoid)
# loss
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(logits),reduction_indices=[1]))
# loss = tf.losses.absolute_difference(labels=ys, predictions=logits)
loss = tf.abs(ys-logits)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
        
saver = tf.train.Saver()
train_data = dataset.CCPD5000_data('./ccpd5000/train/')
train_labels = dataset.CCPD5000_labels('./ccpd5000/train/')
checkpoint_dir = './tmp/ccpd_model_3/'
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(20000):
      data_batch, label_batch = next_batch(32, train_data, train_labels)
      sess.run(train_step, feed_dict={xs: data_batch, ys: label_batch})
      if i % 200 == 0:
          print('loss %d :' % (i))
          print( compute_loss(data_batch, label_batch))
          saver.save(sess, checkpoint_dir + 'model.ckpt')

