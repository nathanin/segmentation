import tensorflow as tf
import numpy as np
import cv2

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

img = cv2.imread('/home/nathan/segmentation/nuclei/data/label/000000.png', -1)
img /= 255
print img.shape, img.min(), img.max(), img.dtype

img = [np.expand_dims(img, 0) for _ in range(3)]
img = np.concatenate(img, 0)

img_tensor = tf.constant(img)
print img_tensor.get_shape(), img_tensor.dtype

img_onehot = tf.one_hot(img_tensor, 2)
print img_onehot.get_shape(), img_onehot.dtype

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    output = sess.run([img_onehot])[0]

    print output.shape, output.dtype
    cv2.imwrite('./c1.jpg', output[0,:,:,0]*255)
    cv2.imwrite('./c2.jpg', output[0,:,:,1]*255)
