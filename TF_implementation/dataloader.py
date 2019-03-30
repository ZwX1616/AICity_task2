import csv
import tensorflow as tf
import numpy as np

# given a batch_size, the dataloader returns
# half positive pairs and half negative pairs
# randomly (when get_batch() is called)
class DataLoader_train:
	# batch_size = 2N
	# data_shape = (H,W,C)
	def __init__(self, batch_size, data_shape):
		assert batch_size%2==0, "batch size should be even numbers"
		self.batch_size = batch_size
		self.data_shape = data_shape
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
		img = tf.image.resize_images(img, [self.data_shape[0],self.data_shape[1]])
		img = img / 255.0;
		# img needs to be .eval().reshape((1,299,299,3)) afterwards
		with tf.Session() as sess:
			return np.array(img.eval().reshape((1,self.data_shape[0],self.data_shape[1],self.data_shape[2])))

	def get_batch(self):
	# returns np.array x1, x2, y
	# x1, x2 with shape (batch, 299, 299, 3)
	# y with shape (batch, )

		# get positive pairs ([B/2*299*299*3, B/2*299*299*3])
		pos_img_left = []
		pos_img_right = []
			# randomly choose batch/2 IDs
		rand_ids = np.random.randint(0,len(self.train_index),int(self.batch_size/2))

		for i in range(int(self.batch_size/2)):
			# randomly choose two images of the same ID
			pair_id = self.train_index[rand_ids[i]][0]+ \
				np.random.choice(self.train_index[rand_ids[i]][1]-self.train_index[rand_ids[i]][0]+1,2)
			# print ("pos pair: " + self.train_filelist[pair_id[0]] + "," + self.train_filelist[pair_id[1]])
			pos_img_left.append(self.load_and_preprocess(self.train_filelist[pair_id[0]]))
			pos_img_right.append(self.load_and_preprocess(self.train_filelist[pair_id[1]]))

		pos_img_left = np.concatenate(pos_img_left,axis=0)
		pos_img_right = np.concatenate(pos_img_right,axis=0)

		# get negative pairs
		neg_img_left = []
		neg_img_right = []

		for i in range(int(self.batch_size/2)):
			# randomly choose 2 IDs
			rand_ids = np.random.choice(len(self.train_index),2)
			# randomly choose an image for each of the IDs
			pair_id = [self.train_index[rand_ids[0]][0]+np.random.randint(0,self.train_index[rand_ids[0]][1]-self.train_index[rand_ids[0]][0]+1),
						self.train_index[rand_ids[1]][0]+np.random.randint(0,self.train_index[rand_ids[1]][1]-self.train_index[rand_ids[1]][0]+1)]
			# print ("neg pair: " + self.train_filelist[pair_id[0]] + "," + self.train_filelist[pair_id[1]])
			neg_img_left.append(self.load_and_preprocess(self.train_filelist[pair_id[0]]))
			neg_img_right.append(self.load_and_preprocess(self.train_filelist[pair_id[1]]))

		neg_img_left = np.concatenate(neg_img_left,axis=0)
		neg_img_right = np.concatenate(neg_img_right,axis=0)

		return np.concatenate((pos_img_left, neg_img_left),axis=0), \
				np.concatenate((pos_img_right, neg_img_right),axis=0), \
				np.concatenate((np.ones(int(self.batch_size/2)),np.zeros(int(self.batch_size/2))),axis=0)