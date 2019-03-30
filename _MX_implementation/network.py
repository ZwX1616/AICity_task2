import mxnet as mx
from mxnet.gluon.model_zoo import vision
from mxnet.gluon import nn

def get_net_inception3(epoch=-1):

    net = nn.HybridSequential()

    # backbone
    feature_net = vision.inception_v3(pretrained=True)
    net.add(feature_net.features)
    net.add(nn.Flatten())

    # freeze parameters
    net.collect_params().setattr('lr_mult', 0)

    # feature mapping 
    # 2048 --> 1024
    net.add(nn.Dense(1024))

    if epoch>=0:
    	# load saved model
        pass
    else:
        net.initialize(mx.init.Xavier())

    net.hybridize()
    return net