import mxnet as mx
import numpy as np

DEFAULT_MEAN = 110
DEFAULT_STD = 58

def get_train_iter(batch_size, data_shape):
	## input:
	# batch_size (int)
	# data_shape (int w,h,c)
	## output:
	# generator with (batch_id, string of %, image_tensor(1/2), image_tensor(2/2), siamese_labels)
	assert batch_size%2==0, 'batch_size should be 2N'

	train_set=np.loadtxt(open("train_label.csv", "rb"), delimiter=",",dtype=str)
	gt_labels=train_set[:,0].astype(int)
	file_list=train_set[:,1]
	del train_set
	num_batches=mx.nd.floor(mx.nd.array([len(file_list)/batch_size])).asscalar().astype(int)

	for i in range(num_batches):
		image_tensor=mx.nd.zeros((batch_size,data_shape[2],data_shape[1],data_shape[0])).astype(float)
		labels=mx.nd.ones((int(batch_size/2))).astype(int)

		for j in range(batch_size):
			img=mx.img.color_normalize(mx.img.imread('./image_train/'+file_list[batch_size*i+j]).astype(float)
				,DEFAULT_MEAN,DEFAULT_STD)
			img_resized=mx.img.imresize(img,data_shape[0],data_shape[1])
			image_tensor[j,:,:,:]=img_resized.swapaxes(2,1).swapaxes(1,0)

			if j<batch_size/2:
				if gt_labels[batch_size*i+j]!=gt_labels[batch_size*i+j+int(batch_size/2)]:
					labels[j]=-1 # -1 for cosine loss, 0 for contrastive loss

		progress='{:.2f}%'.format(100*i/num_batches)
		yield i, progress, image_tensor[:int(batch_size/2)], image_tensor[int(batch_size/2):], labels