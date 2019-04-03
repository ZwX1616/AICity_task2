import tensorflow as tf
import numpy as np
from time import time

from network import Siamese_classic_mobilenet
from dataloader import DataLoader_eval

batch_size = 16
data_shape = (224,224,3)
data_set = 'test'

sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
# define network
net = Siamese_classic_mobilenet()
# load trained model
saver = tf.train.Saver()
saver.restore(sess, './checkpoints/model')
with open('./checkpoints/iter.txt','r+') as f:
	epoch_offset = int(f.read())+1
print ('...checkpoint loaded from iteration ' + \
		str(epoch_offset) + \
		'...')

# dataloader for evaluation
my_dataloader = DataLoader_eval(batch_size, data_shape, data_set)

outputs = []
while(my_dataloader.is_complete==False):
	x = my_dataloader.get_batch()
	y = sess.run(net.o1, feed_dict={net.x1:x})
	outputs.append(y)

outputs = np.concatenate(outputs,axis=0)
print('output shape: '+str(outputs.shape)+'. saving to ./output/'+data_set+'_feat_vec.npy')
np.save('./output/'+data_set+'_feat_vec.npy', outputs)