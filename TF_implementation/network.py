import tensorflow as tf
from tensorflow.keras.models import Model

class Siamese_classic_inception3:

    def __init__(self):
        from tensorflow.keras.applications import inception_v3
        # define input image pair
        self.x1 = tf.placeholder(tf.float32, [None, 299, 299, 3])
        self.x2 = tf.placeholder(tf.float32, [None, 299, 299, 3])

        # define network
        with tf.variable_scope("siamese") as scope:
            self.backbone_model = inception_v3.InceptionV3(weights='imagenet')
            self.bottleneck_model = Model(inputs=self.backbone_model.input,
                    outputs=self.backbone_model.get_layer('avg_pool').output)
#            for layer in self.bottleneck_model.layers:
#                layer.trainable = False 
                
            self.bottleneck_feature_1=self.bottleneck_model(self.x1)
            self.bottleneck_feature_2=self.bottleneck_model(self.x2)
            
            self.o1 = self.feature_vector_mapping(self.bottleneck_feature_1)
            scope.reuse_variables()
            self.o2 = self.feature_vector_mapping(self.bottleneck_feature_2)
    
        # define loss
        self.y_gt = tf.placeholder(tf.float32, [None]) # 1 or 0
        self.loss = self.loss_function()

    def feature_vector_mapping(self, feat_vec):
        # this is our trainable metric mapping function
        # add a FC layer 2048 -> 768
        fc_output = self.fc_layer(feat_vec, 768, "feat_vec_mapping")

        return fc_output

    def fc_layer(self, input, output_len, name):
#        print (input.get_shape())
        input_len = input.get_shape()[1]

        # ac = tf.nn.relu(..)
        W = tf.get_variable(name+'_W', dtype=tf.float32, shape=[input_len, output_len], 
                            initializer=tf.keras.initializers.glorot_normal,
                            regularizer=tf.contrib.layers.l2_regularizer(0.005))
        b = tf.get_variable(name+'_b', dtype=tf.float32, shape=[output_len],
                            initializer=tf.zeros_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(0.005))
        fc = tf.nn.bias_add(tf.matmul(input, W), b)

        return fc

    def loss_function(self):
        # cosine similarity loss
        # o1, o2 format: [batch x 768]
        o1_normalized = tf.nn.l2_normalize(self.o1,1) #axis=1
        o2_normalized = tf.nn.l2_normalize(self.o2,1)
        cos_sim=tf.add(1.0, tf.reduce_sum(tf.multiply(o1_normalized,o2_normalized),axis=1), name="cos_sim")

#         1 -> pos, 0 -> neg
#         print(cos_sim.get_shape())
        neg = tf.multiply(tf.subtract(1.0, self.y_gt, name="neg_mask"), cos_sim, name="neg_loss")
        pos = tf.multiply(self.y_gt, tf.subtract(2.0, cos_sim, name="pos_loss_0"), name="pos_loss")
        total_loss = tf.add(neg, pos, name="total_loss")
        loss = tf.reduce_mean(total_loss, name="loss")

        return loss

class Siamese_classic_mobilenet:

    def __init__(self):
        from tensorflow.keras.applications import mobilenet
        # define input image pair
        self.x1 = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.x2 = tf.placeholder(tf.float32, [None, 224, 224, 3])

        # define network
        with tf.variable_scope("siamese") as scope:
            self.backbone_model = mobilenet.MobileNet()
            self.bottleneck_model = Model(inputs=self.backbone_model.input,
                    outputs=self.backbone_model.get_layer('global_average_pooling2d').output)
#            for layer in self.bottleneck_model.layers:
#                layer.trainable = False 
                
            self.bottleneck_feature_1=self.bottleneck_model(self.x1)
            self.bottleneck_feature_2=self.bottleneck_model(self.x2)
            
            self.o1 = self.feature_vector_mapping(self.bottleneck_feature_1)
            scope.reuse_variables()
            self.o2 = self.feature_vector_mapping(self.bottleneck_feature_2)
    
        # define loss
        self.y_gt = tf.placeholder(tf.float32, [None]) # 1 or 0
        self.loss = self.loss_function()

    def feature_vector_mapping(self, feat_vec):
        # this is our trainable metric mapping function
        # add a FC layer 2048 -> 768
        fc_output = self.fc_layer(feat_vec, 256, "feat_vec_mapping")

        return fc_output

    def fc_layer(self, input, output_len, name):
#        print (input.get_shape())
        input_len = input.get_shape()[1]

        # ac = tf.nn.relu(..)
        W = tf.get_variable(name+'_W', dtype=tf.float32, shape=[input_len, output_len], 
                            initializer=tf.keras.initializers.glorot_normal,
                            regularizer=tf.contrib.layers.l2_regularizer(0.005))
        b = tf.get_variable(name+'_b', dtype=tf.float32, shape=[output_len],
                            initializer=tf.zeros_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(0.005))
        fc = tf.nn.bias_add(tf.matmul(input, W), b)

        return fc

    def loss_function(self):
        # cosine similarity loss
        # o1, o2 format: [batch x 768]
        o1_normalized = tf.nn.l2_normalize(self.o1,1) #axis=1
        o2_normalized = tf.nn.l2_normalize(self.o2,1)
        cos_sim=tf.add(1.0, tf.reduce_sum(tf.multiply(o1_normalized,o2_normalized),axis=1), name="cos_sim")

#         1 -> pos, 0 -> neg
#         print(cos_sim.get_shape())
        neg = tf.multiply(tf.subtract(1.0, self.y_gt, name="neg_mask"), cos_sim, name="neg_loss")
        pos = tf.multiply(self.y_gt, tf.subtract(2.0, cos_sim, name="pos_loss_0"), name="pos_loss")
        total_loss = tf.add(neg, pos, name="total_loss")
        loss = tf.reduce_mean(total_loss, name="loss")

        return loss

if __name__ == '__main__':
#    import pdb; pdb.set_trace() ###
    # for debug purposes
    with tf.Session() as sess:
        # this has to be put inside the session context
#         net = Siamese_classic_inception3()
#         import numpy as np
#         x1 = np.ones((3,299,299,3))
#         x2 = np.ones((3,299,299,3))
#         x1[2] = - x1[0]
#         x2[1] = - x1[0]
#         x2[2] = - x1[0]
#         y = np.array([1,0,1])
#         y_same = np.ones(1)
#         y_diff = np.zeros(1)
        
#         # initialize the custom layer weights 
#         uninitialized_var = [v for v in tf.global_variables() if "feat_vec_mapping" in v.name]
#         print ("uninitialized variables: " + str(len(uninitialized_var)))
#         tf.variables_initializer(uninitialized_var).run()
  
#         loss1 = sess.run(net.loss, feed_dict = {net.x1: x1, net.x2: x2, net.y_gt: y})
# #        loss2 = sess.run(net.loss, feed_dict = {net.x1: x1, net.x2: x1, net.y_gt: y_diff})
# #        loss3 = sess.run(net.loss, feed_dict = {net.x1: x1, net.x2: x2, net.y_gt: y_same})
# #        loss4 = sess.run(net.loss, feed_dict = {net.x1: x1, net.x2: x2, net.y_gt: y_diff})
#         import pdb; pdb.set_trace()
#         file_writer = tf.summary.FileWriter('./graph/')
#         file_writer.add_graph(sess.graph)
#         file_writer.close()
#         print("loss1="+str(loss1)+", loss2="+str(loss2)+", loss3="+str(loss3)+", loss4="+str(loss4))
# tensorboard --logdir="C:\Users\weixing\Documents\code\Nvidia_AIC_2019\AICity_task2\TF_implementation\graph_mn" --host=127.0.0.1
        from tensorflow.keras.applications import mobilenet
        net = mobilenet.MobileNet()
        import numpy as np
        x = np.ones((1,224,224,3))
        print(net(x.astype('float32')).eval())
        file_writer = tf.summary.FileWriter('./graph_mn/')
        file_writer.add_graph(sess.graph)
        file_writer.close()