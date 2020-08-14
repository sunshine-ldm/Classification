import dataset
import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np
from numpy.random import seed

seed(10)
tf.set_random_seed(20)
# tf.random.set_seed(20)
batch_size = 500
classes = ['cat', 'dog']
num_classes = len(classes)
test_size = 0.2
img_size = 64
num_channels = 3
train_path = 'data/train'
data = dataset.read_train_sets(train_path, img_size, classes, test_size=test_size)
print("数据读取完毕")
print("训练集的数量为：{}".format(len(data.train._images)))
print("测试集的数量为：{}".format(len(data.test._images)))
session = tf.Session()    # 2.0版本没有Session模块

# tf.compat.v1.disable_eager_execution()

x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
# labels
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, 1)

filter_size_conv1 = 3
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 32

filter_size_conv3 = 3
num_filters_conv3 = 64

fc_layer_size = 1024


def create_weights(shape):
    # return tf.Variable(tf.random.truncated_normal(shape, stddev=0.05))  # 截断正态分布
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


def create_conv_layer(input, num_input_channels, conv_filter_size, num_filters):
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    biases = create_biases(num_filters)
    # layer = tf.nn.conv2d(input=input, filters=weights,
    #                      strides=[1, 1, 1, 1], padding='SAME')
    layer = tf.nn.conv2d(input=input, filter=weights,
                         strides=[1, 1, 1, 1], padding='SAME')
    layer += biases
    layer = tf.nn.relu(layer)

    # layer = tf.nn.max_pool(input=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return layer


def create_flatten_layer(layer):  # 将卷积后的特征图拉伸平铺
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()  # 计算总的特征数
    layer = tf.reshape(layer, [-1, num_features])  # reshape成(1,num_features)
    return layer


def create_fc_layer(input, num_inputs, num_outputs, use_relu=True):
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)
    layer = tf.matmul(input, weights) + biases  # matmul()返回两个矩阵相乘的结果
    # layer = tf.nn.dropout(layer, keep_prob=0.7)
    layer = tf.nn.dropout(layer, 0.7)
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer


layer_conv1 = create_conv_layer(x, num_channels, filter_size_conv1, num_filters_conv1)
layer_conv2 = create_conv_layer(layer_conv1, num_filters_conv1, filter_size_conv2, num_filters_conv2)
layer_conv3 = create_conv_layer(layer_conv2, num_filters_conv2, filter_size_conv3, num_filters_conv3)

layer_flat = create_flatten_layer(layer_conv3)  # 在全连接层进行之前拉伸特征图
layer_fc1 = create_fc_layer(layer_flat, layer_flat.get_shape()[1:4].num_elements(),
                            fc_layer_size, use_relu=True)
layer_fc2 = create_fc_layer(layer_fc1, fc_layer_size, num_classes, use_relu=False)

y_perd = tf.nn.softmax(layer_fc2, name='y_pred')
y_perd_cls = tf.argmax(y_perd, 1)

session.run(tf.global_variables_initializer())  # 全局变量初始化
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimize(learning_rate=1e-3).minimize(cost)  # tf1.x 版本中使用方式
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(1e-3).minimize(cost)
# optimizer = tf.optimizers.Adam(1e-3)
correct_prediction = tf.equal(y_perd_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session.run(tf.global_variables_initializer())


def show_progress(epoch, feed_dict_train, feed_dict_test, val_loss, i):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_test)
    msg = "训练周期：{0}--- 迭代次数：{1}--- 训练精度：{2:>6.1%},测试精度：{3:>6.1%},测试损失：{4}"
    print(msg.format(epoch + 1, i, acc, val_acc, val_loss))


total_iter = 0
saver = tf.train.Saver()


def train(num_iter):
    global total_iter
    for i in range(total_iter, total_iter + num_iter):
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_test_batch, y_test_batch, _, test_cls_batch = data.test.next_batch(batch_size)
        feed_dict_tr = {x: x_batch, y_true: y_true_batch}
        feed_dict_val = {x: x_test_batch, y_true: y_test_batch}
        session.run(optimizer, feed_dict=feed_dict_tr)

        if i % int(data.train.num_examples / batch_size) == 0:  # 一个batch中有多少样本
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train.num_examples / batch_size))
            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss, i)
            saver.save(session, 'model/cat_dog.ckpt', global_step=i)
    total_iter += num_iter


train(num_iter=8000)

