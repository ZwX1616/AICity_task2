import csv
import tensorflow as tf
import numpy as np

# given a batch_size, the dataloader returns
# half positive pairs and half negative pairs
# randomly (when asked to)
class DataLoader_train:

	def __init__(self, batch_size):
		assert batch_size%2==0, "batch size should be even numbers"
		self.batch_size = batch_size

		# load the indexes
		self.train_index = []
		with open('./data/train_index.txt',encoding='utf-8') as cfile:
			reader = csv.reader(cfile)
			readeritem=[]
			readeritem.extend([row for row in reader])
		for _, row in enumerate(readeritem):
			self.train_index.append([int(row[0]),int(row[1])])
		del reader
		del readeritem

		# load the filelist
		self.train_filelist = []
		with open('./data/train_label.csv',encoding='utf-8') as cfile:
			reader = csv.reader(cfile)
			readeritem=[]
			readeritem.extend([row for row in reader])
		for _, row in enumerate(readeritem):
			self.train_filelist.append(row[1])
		del reader
		del readeritem

	def load_and_preprocess(self, image_file):
		img = tf.read_file('./data/image_train/'+image_file)
		img = tf.image.decode_jpeg(img)
		img = tf.image.resize_images(img, [299,299])
		img = img / 255.0;
		# img needs to be .eval().reshape((1,299,299,3)) afterwards
		with tf.Session() as sess:
			return np.array(img.eval().reshape((1,299,299,3)))

	def get_batch(self):
		pos_img_left = []
		pos_img_right = []
		# get positive pairs ([B/2*299*299*3, B/2*299*299*3])
			# random identity
		rand_ids = np.random.randint(0,len(self.train_index),int(self.batch_size/2))
		for i in range(int(self.batch_size/2)):
			pair_id = self.train_index[rand_ids[i]][0]+ \
				np.random.choice(self.train_index[rand_ids[i]][1]-self.train_index[rand_ids[i]][0]+1,2)
			import pdb; pdb.set_trace() ###
			pos_img_left.append([load_and_preprocess(self.train_filelist[pair_id[0]]),
			pos_img_right
							load_and_preprocess(self.train_filelist[pair_id[1]])])


		# get negative pairs