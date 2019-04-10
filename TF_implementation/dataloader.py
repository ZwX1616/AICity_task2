import csv
import tensorflow as tf
import numpy as np
from scipy import misc

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
		img = misc.imread('./data/image_train/'+image_file)
		img = misc.imresize(img,(self.data_shape[0],self.data_shape[1]))
		img = img / 255.0;
		return img.reshape((1,self.data_shape[0],self.data_shape[1],self.data_shape[2]))

	def get_batch(self):
	# returns np.array x1, x2, y
	# x1, x2 with shape (batch, H, W, C)
	# y with shape (batch, )

		# get positive pairs ([B/2*H*W*C, B/2*H*W*C])
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
			rand_ids = np.random.choice(len(self.train_index),2,replace=False)
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


# load batch_size images and return them
# for forward computation
class DataLoader_eval():
	# batch_size = N
	# data_shape = (H,W,C)
	# data_set = 'test' or 'query'
	def __init__(self, batch_size, data_shape, data_set):
		self.batch_size = batch_size
		self.data_shape = data_shape
		assert data_set in ['query','test'], "data_set should be 'test' or 'query'"
		self.data_set = data_set

		# load the filelist
		self.eval_filelist = []
		with open('./data/name_'+self.data_set+'.txt',encoding='utf-8') as cfile:
			reader = csv.reader(cfile)
			readeritem=[]
			readeritem.extend([row for row in reader])
		for _, row in enumerate(readeritem):
			self.eval_filelist.append(row[0])
		del reader
		del readeritem

		# initialize index
		self.current_batch = 0
		self.is_complete = False

	def load_and_preprocess(self, image_file):
		img = misc.imread('./data/image_'+self.data_set+'/'+image_file)
		img = misc.imresize(img,(self.data_shape[0],self.data_shape[1]))
		img = img / 255.0;
		return img.reshape((1,self.data_shape[0],self.data_shape[1],self.data_shape[2]))

	def reset_batch(self):
		self.current_batch = 0
		self.is_complete = False

	def get_batch(self):
	# returns np.array x
	# x with shape (batch, H, W, C)
		if (self.is_complete==True):
			print ("have finished reading all batches. please reset_batch().")
			return []

		img = []
		if (len(self.eval_filelist)-self.batch_size*(self.current_batch+1)>0):
		# remaining images more than batch_size
			for i in range(self.batch_size):
				img.append(self.load_and_preprocess(self.eval_filelist[self.batch_size*self.current_batch+i]))
			self.current_batch = self.current_batch + 1
		else:
		# remaining images less than or equal batch_size (last batch)
			for i in range(len(self.eval_filelist)-self.batch_size*self.current_batch):
				img.append(self.load_and_preprocess(self.eval_filelist[self.batch_size*self.current_batch+i]))
			self.current_batch = self.current_batch + 1
			self.is_complete=True

		return np.concatenate(img,axis=0)

# iterate through all the query images
# for each query image, load batch_size images from the test set \
# and compute the matching score
class DataLoader_eval_v2():
	# batch_size = N
	# data_shape = (H,W,C)
	def __init__(self, batch_size, data_shape):
		self.batch_size = batch_size
		self.data_shape = data_shape

		# load the query filelist
		self.query_filelist = []
		with open('./data/name_query.txt',encoding='utf-8') as cfile:
			reader = csv.reader(cfile)
			readeritem=[]
			readeritem.extend([row for row in reader])
		for _, row in enumerate(readeritem):
			self.query_filelist.append(row[0])
		del reader
		del readeritem

		# load the test filelist
		self.test_filelist = []
		with open('./data/name_test.txt',encoding='utf-8') as cfile:
			reader = csv.reader(cfile)
			readeritem=[]
			readeritem.extend([row for row in reader])
		for _, row in enumerate(readeritem):
			self.test_filelist.append(row[0])
		del reader
		del readeritem

		# initialize query index
		self.current_query = 0
		self.complete_all = False
		# initialize test set index
		self.current_batch = 0
		self.is_complete = False

	def load_and_preprocess(self, image_file, data_set):
		img = misc.imread('./data/image_'+data_set+'/'+image_file)
		img = misc.imresize(img,(self.data_shape[0],self.data_shape[1]))
		img = img / 255.0;
		return img.reshape((1,self.data_shape[0],self.data_shape[1],self.data_shape[2]))

	def reset_batch(self):
		self.current_query = self.current_query + 1
		self.current_batch = 0
		self.is_complete = False

	def get_batch(self):
	# returns np.array x1, x2
	# x1 is the same query image repeated, with shape (batch, H, W, C)
	# x2 is batch_size different test images, with shape (batch, H, W, C)
		if (self.is_complete==True):
			self.reset_batch()

		if (self.current_query==len(self.query_filelist)):
			print ("have finished reading all query images. thank you for choosing our service")
			self.complete_all = True
			return []

		# img_query = []
		img_query = self.load_and_preprocess(self.query_filelist[self.current_query],"query")

		img_test = []
		if (len(self.test_filelist)-self.batch_size*(self.current_batch+1)>0):
		# remaining images more than batch_size
			for i in range(self.batch_size):
				img_test.append(self.load_and_preprocess(self.test_filelist[self.batch_size*self.current_batch+i],"test"))
			self.current_batch = self.current_batch + 1
		else:
		# remaining images less than or equal batch_size (last batch)
			for i in range(len(self.test_filelist)-self.batch_size*self.current_batch):
				img_test.append(self.load_and_preprocess(self.test_filelist[self.batch_size*self.current_batch+i],"test"))
			self.current_batch = self.current_batch + 1
			self.is_complete=True

		return np.tile(img_query,(len(img_test),1,1,1)), \
				np.concatenate(img_test,axis=0)