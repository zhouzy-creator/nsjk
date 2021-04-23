import cv2, os, numpy
import tensorflow as tf

#if you have stored the images in a directory, directory_path is the absolute path to the directory

imgs = []
for file_ in os.listdir(directory_path):
img = cv2.imread(os.path.join(directory_path,file_))
imgs.append(img)

#all the images must be of same size, you will have to resize if they are not of same size
#imgs is list of numpy ndarrays, you can convert this list to array so that now this array can be used as a tensor

imgs_array = numpy.array(img)

#You will have defined an op called 'pred'
#Now when you sess.run the op, you will use feed_dict to feed the imgs_array tensor
with tf.Session() as sess:
preds = sess.run(pred, feed_dict={imgs_ph:imgs_array})