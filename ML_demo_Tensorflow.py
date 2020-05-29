import h5py
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto


def batch_norm(x, n_out, phase_train, name):
    with tf.device('/GPU:0'):
        with tf.variable_scope('bn'):
            beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                               name='beta'+str(name), trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                name='gamma'+str(name), trainable=True)
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments'+str(name))
            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(phase_train,
                                mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


class Net(object):

    def NetModel(self, images, labels, phase_train):
        conv_1_out = tf.layers.conv2d(name=None, inputs=images,
                                      kernel_size=[3, 3], filters=32,
                                      padding='SAME', activation=tf.nn.relu)

        conv_1_out = batch_norm(conv_1_out, 32, phase_train=phase_train, name='_0')
        conv_1_out = tf.nn.relu(conv_1_out)
        conv_1_out = tf.nn.max_pool(conv_1_out, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        conv_2_out = tf.layers.conv2d(name=None, inputs=conv_1_out,
                                      kernel_size=[3, 3], filters=32,
                                      padding='SAME', activation=tf.nn.relu)

        conv_2_out = batch_norm(conv_2_out, 32, phase_train=phase_train, name='_1')
        conv_2_out = tf.nn.relu(conv_2_out)
        conv_2_out = tf.nn.max_pool(conv_2_out, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        flatten_input = tf.layers.flatten(conv_2_out)
        fc1 = tf.layers.dense(flatten_input, 1024)
        fc2 = tf.layers.dense(fc1, 10)
        output = fc2
        return output, labels,


images = tf.placeholder(tf.float32, [None, 256, 256, 3], name='images')
labels = tf.placeholder(tf.float32, [None, 1], name='labels')
phase_train = tf.placeholder(tf.bool, name='phase_train')
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(.001, global_step, 5000, 0.2295, staircase=True)

net = Net()
outputs, labels = net.NetModel(images=images, labels=labels, phase_train=phase_train)

loss_= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=outputs))
tf.compat.v1.summary.scalar("loss", loss_)
optimizer3 = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-8,
                                             beta1=0.9,
                                             beta2=.999,
                                             use_locking=True).minimize(loss_,global_step=global_step, name='optimizer')
merged_summary_op = tf.compat.v1.summary.merge_all()

init = tf.global_variables_initializer()
config = ConfigProto()
sess = tf.Session(config=config)
sess.run(init)

loss_orig_all = []
loss_body_all = []

f = h5py.File('/home/iman/myexperiments/pipe_segmentation/dataset/data_demo.h5', 'r')
images_h5 = np.asarray(list(f['img']))
labels_h5 = np.asarray(list(f['labels']))




loss_train_all = []
loss_train_body_all = []
loss_test_all = []
loss_test_body_all = []
UV_map_acc_all = []
I_map_acc_all = []
fid_all = []
# summary_writer = tf.summary.FileWriter('./', graph=tf.get_default_graph())
# merged_summary_op = tf.compat.v1.summary.merge_all()
import time
t = time.time()
for i_iter in range(0, 4):
    with tf.device('/GPU:0'):
        images_batch = images_h5[i_iter*25:(i_iter+1)*25]
        labels_batch = labels_h5[i_iter*25:(i_iter+1)*25]

        _, _, loss_orig = sess.run([outputs, optimizer3, loss_],#, merged_summary_op],
                                            feed_dict={images: images_batch,
                                                       labels: labels_batch,
                                                       phase_train:True})

e = time.time()
print('Time for training the network for 4 iterations with batch 25:', e-t)
import ipdb;

ipdb.set_trace()
