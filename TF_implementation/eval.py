import tensorflow as tf
import numpy as np
from time import time

from network import Siamese_typeC_CE_loss
from dataloader import DataLoader_eval_v2

batch_size = 256
data_shape = (128,128,3)

top_k_number = 100

sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
# define network
net = Siamese_typeC_CE_loss(training=False)
# load trained model
saver = tf.train.Saver()
saver.restore(sess, './checkpoints/model')
with open('./checkpoints/iter.txt','r+') as f:
	epoch_offset = int(f.read())+1
print ('...checkpoint loaded from iteration ' + \
		str(epoch_offset) + \
		'...')

# dataloader for evaluation
my_dataloader = DataLoader_eval_v2(batch_size, data_shape)
#all_scores = [] # one row is one query, with each col correspond to one test
current_scores = []
import csv
while(my_dataloader.complete_all==False):
	print("evaluation progress - overall:{:.2f}%, ".format(100*my_dataloader.current_query/len(my_dataloader.query_filelist)),
			"current query:{:.2f}%".format(min(100*my_dataloader.batch_size*(my_dataloader.current_batch)/len(my_dataloader.test_filelist),100)),
			end='\r')
	try: 
		x1,x2 = my_dataloader.get_batch()
	except:
		break
	y = sess.run(net.output, feed_dict={net.x1:x1, net.x2:x2})
	current_scores.extend(y[:,1].tolist())
	if my_dataloader.is_complete==True:
		current_ranking = np.flip(np.argsort(np.array(current_scores)),axis=0)[:min(top_k_number, len(current_scores))] + 1
		with open('./output/submission_hey.txt', 'a+', newline='') as wf:
			writer = csv.writer(wf)
			writer.writerow([' '.join(str(c) for c in current_ranking)])
		print('output shape: ' + str(len(current_scores)) + '. saved to ./output/submission_hey.txt', '\n', 'good luck!')
		current_scores = []
