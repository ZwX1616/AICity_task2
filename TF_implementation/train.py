import tensorflow as tf
import numpy as np
from time import time

from network import Siamese_classic_mobilenet
from dataloader import DataLoader_train

learning_rate = 0.1
num_iter = 10000
batch_size = 256

save_interval = 25

data_shape = (224,224,3)

#config = tf.ConfigProto()

# operate within tf session
sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))

# setup networkse
# by default this use pretrained weights of incepion3
# as our backbone
net = Siamese_classic_mobilenet()

# saver.restore(sess, './model')
# initialize the custom layer weights 
trainable_var = [v for v in tf.global_variables() if "feat_vec_mapping" in v.name]
print ("trainable variables: ")
print (trainable_var)
tf.variables_initializer(trainable_var).run()

# setup trainer
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss=net.loss, 
											  var_list=trainable_var)
my_dataloader = DataLoader_train(batch_size, data_shape)

# variable saver
saver = tf.train.Saver()

# tensorboard logger
writer_train = tf.summary.FileWriter('./logs/')
tf.summary.scalar("net_loss", net.loss)
merged_summary = tf.summary.merge_all()

# iterate
for epoch in range(num_iter):
	start = time()
	x1, x2, y = my_dataloader.get_batch()
	_, summary, l = sess.run([train_step, merged_summary, net.loss], feed_dict={
						net.x1: x1,
						net.x2: x2,
						net.y_gt: y})

	if np.isnan(l):
		print('Model diverged with loss = NaN')
		quit()

	# logging
	print ('iteration %d: loss %.7f, runtime: %.1fs' % (epoch, l, time()-start))
	if (epoch==0):
		writer_train.add_graph(sess.graph)
	writer_train.add_summary(summary, epoch)
	writer_train.flush()
	if (epoch>0 and epoch%save_interval==0):
		saver.save(sess, './checkpoints/model')
		print ('...checkpoint saved...')

print (str(num_iter)+' iterations completed')
writer_train.close()
sess.close()