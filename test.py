import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse

img_size = 64
num_channels = 3
images = []
path = 'data/test/36.jpg'
image = cv2.imread(path)
image = cv2.resize(image, (img_size, img_size), 0, 0,cv2.INTER_LINEAR)
images.append(image)
images = np.array(images, dtype=np.uint8)
images = images.astype('float32')
images = np.multiply(images,1.0/255.0)
x_batch = images.reshape(1,img_size,img_size,num_channels)
sess = tf.Session()
# tf.compat.v1.disable_eager_execution()
saver = tf.train.import_meta_graph('model/cat_dog.ckpt-7840.meta')   # 选取训练好的模型测试
saver.restore(sess,'model/cat_dog.ckpt-7840')
graph = tf.get_default_graph()
y_pred = graph.get_tensor_by_name("y_pred:0")
x = graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y_true:0")
y_test_img = np.zeros((1,2))

feed_dict_test = {x:x_batch, y_true:y_test_img}
result = sess.run(y_pred, feed_dict=feed_dict_test)

res_label = ['cat', 'dog']
print(res_label[result.argmax()])
