# train a vehicle re-id Siamese network

import mxnet as mx
from mxnet import autograd, gluon
from time import time

from network import get_net_inception3
from dataloader import get_train_iter

# ctx=mx.cpu()
ctx=mx.gpu(0)

batch_size=2 # images per batch, should be 2*n (in pairs)
data_shape=(300,300,3) # w,h,c

num_epochs=20
lr=0.001
lr_period=10
lr_decay=0.1
wd=0.001

# loss
loss = mx.gluon.loss.CosineEmbeddingLoss()

def train(net, num_epochs, learning_rate, weight_decay, batch_size):
    # num_epochs, learning_rate, weight_decay, batch_size
    trainer = mx.gluon.Trainer(net.collect_params(), 'sgd', {
        'learning_rate': learning_rate, 'momentum': 0.9, 'wd': weight_decay})

    for epoch in range(num_epochs):
        if epoch == 0:
            print ('epoch='+str(epoch)+' lr='+str(trainer.learning_rate))
        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
            print ('epoch='+str(epoch)+' lr='+str(trainer.learning_rate))

        start=time()
        train_iter = get_train_iter(batch_size, data_shape) # training data loader
        for _, progress, X1, X2, y in train_iter:
            with autograd.record():
                l = loss(net(X1.astype('float32')),net(X2.astype('float32')),y.astype('float32'))
            l.backward()
            # train_ls.append(l.mean().asscalar())
            trainer.step(batch_size)
        print('epoch '+str(epoch)+' done. abs='+str(l.mean().asscalar())+ ' time={:.2f}s'.format(time()-start))#,end='\r')

if __name__=='__main__':

    net = get_net_inception3(-1) # specify saved checkpoint>=0

    train(net, num_epochs, lr, wd, batch_size)