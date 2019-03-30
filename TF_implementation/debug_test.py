import tensorflow as tf
from network import Siamese_classic_inception3
with tf.Session() as sess:
	# load network
	net = Siamese_classic_inception3()

	# load trained parameters
	saver = tf.train.Saver()
	saver.restore(sess, './model')

	# load dummy images for testing
	test_img_1 = tf.read_file('test_1.jpg')
	test_img_1 = tf.image.decode_jpeg(test_img_1)
	test_img_1 = tf.image.resize_images(test_img_1, [299,299])
	test_img_1 = test_img_1 / 255.0;
	test_img_1 = test_img_1.eval().reshape((1,299,299,3))
	test_img_2 = tf.read_file('test_2.jpg')
	test_img_2 = tf.image.decode_jpeg(test_img_2)
	test_img_2 = tf.image.resize_images(test_img_2, [299,299])
	test_img_2 = test_img_2 / 255.0;
	test_img_2 = test_img_2.eval().reshape((1,299,299,3))

	f11, f12 = sess.run([net.o1, net.o2], feed_dict={
                        net.x1: test_img_1,
                        net.x2: test_img_1})
	f21, f22 = sess.run([net.o1, net.o2], feed_dict={
                        net.x1: test_img_2,
                        net.x2: test_img_2})
	import pdb; pdb.set_trace() ###
	print("wdf")