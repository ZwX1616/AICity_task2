import tensorflow as tf
import numpy as np

from network import Siamese_classic_inception3

#config = tf.ConfigProto()

# operate within tf session
sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))

# setup networkse
# by default this use pretrained weights of incepion3
# as our backbone
net = Siamese_classic_inception3()

# saver.restore(sess, './model')
# initialize the custom layer weights 
trainable_var = [v for v in tf.global_variables() if "feat_vec_mapping" in v.name]
print ("trainable variables: ")
print (trainable_var)
tf.variables_initializer(trainable_var).run()

# setup trainer
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss=net.loss, 
                                              var_list=trainable_var)

# variable saver
saver = tf.train.Saver()

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

x1 = np.concatenate((test_img_1,test_img_1,test_img_2), axis=0)
x2 = np.concatenate((test_img_1,test_img_2,test_img_2), axis=0)
y = np.array([1,0,1])

for epoch in range(100):

#    if (epoch%3==0):
#        # 1 1
#        x1 = test_img_1
#        x2 = test_img_1
#        y = np.ones(1)
#    if (epoch%3==1):
#        # 1 2
#        x1 = test_img_1
#        x2 = test_img_2
#        y = np.zeros(1)
#    if (epoch%3==2):
#        # 2 2
#        x1 = test_img_2
#        x2 = test_img_2
#        y = np.ones(1)
        
    _, l = sess.run([train_step, net.loss], feed_dict={
                        net.x1: x1,
                        net.x2: x2,
                        net.y_gt: y})

    if np.isnan(l):
        print('Model diverged with loss = NaN')
        quit()

    print ('epoch %d: loss %.6f' % (epoch, l))

saver.save(sess, './model')
sess.close()