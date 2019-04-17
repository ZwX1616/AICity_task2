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
                            initializer=tf.keras.initializers.random_normal,
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
        # add a FC layer 1024 -> 256
        fc_output = self.fc_layer(feat_vec, 256, "feat_vec_mapping")

        return fc_output

    def fc_layer(self, input, output_len, name):
#        print (input.get_shape())
        input_len = input.get_shape()[1]

        # ac = tf.nn.relu(..)
        W = tf.get_variable(name+'_W', dtype=tf.float32, shape=[input_len, output_len], 
                            initializer=tf.keras.initializers.random_normal,
                            regularizer=tf.contrib.layers.l2_regularizer(0.005))
        b = tf.get_variable(name+'_b', dtype=tf.float32, shape=[output_len],
                            initializer=tf.zeros_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(0.005))
        fc = tf.nn.bias_add(tf.matmul(input, W), b)

        return fc

    def loss_function(self):
        # cosine similarity loss
        # o1, o2 format: [batch x 256]
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


class Siamese_classic_mobilenet_CE_loss:

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

            self.bottleneck_feature_1 = self.bottleneck_model(self.x1)
            scope.reuse_variables()
            self.bottleneck_feature_2 = self.bottleneck_model(self.x2)

            self.o1 = self.bottleneck_feature_1
            self.o2 = self.bottleneck_feature_2

        # define loss
        self.y_gt = tf.placeholder(tf.int32, [None])  # 1 or 0
        self.loss = self.loss_function()

    def fc_layer(self, input, output_len, name):
#        print (input.get_shape())
        input_len = input.get_shape()[1]

        # ac = tf.nn.relu(..)
        W = tf.get_variable(name+'_W', dtype=tf.float32, shape=[input_len, output_len], 
                            initializer=tf.keras.initializers.random_normal,
                            regularizer=tf.contrib.layers.l2_regularizer(0.005))
        b = tf.get_variable(name+'_b', dtype=tf.float32, shape=[output_len],
                            initializer=tf.zeros_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(0.005))
        fc = tf.nn.bias_add(tf.matmul(input, W), b)

        return fc

    def loss_function(self):
        # o1, o2 format: [batch x 1024]
            # compute the L1 difference of two features
        feat_diff = tf.subtract(self.o1, self.o2, name="feat_diff")
        feat_diff_abs = tf.abs(feat_diff, name="feat_diff_abs")
            # FC mapping 1024 -> 2 for binary classification
        fc_output = self.fc_layer(feat_diff_abs, 2, "fc_1024_to_2")
            # convert y_gt to one_hot encoding
        y_gt_onehot = tf.one_hot(self.y_gt, 2)
            # compute 
        loss = tf.losses.softmax_cross_entropy(y_gt_onehot, fc_output)
        return loss

class Siamese_classic_mobilenet_CE_loss:

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

            self.bottleneck_feature_1 = self.bottleneck_model(self.x1)
            scope.reuse_variables()
            self.bottleneck_feature_2 = self.bottleneck_model(self.x2)

            self.o1 = self.bottleneck_feature_1
            self.o2 = self.bottleneck_feature_2

        # define loss
        self.y_gt = tf.placeholder(tf.int32, [None])  # 1 or 0
        self.loss = self.loss_function()

    def fc_layer(self, input, output_len, name):
#        print (input.get_shape())
        input_len = input.get_shape()[1]

        # ac = tf.nn.relu(..)
        W = tf.get_variable(name+'_W', dtype=tf.float32, shape=[input_len, output_len], 
                            initializer=tf.keras.initializers.random_normal,
                            regularizer=tf.contrib.layers.l2_regularizer(0.005))
        b = tf.get_variable(name+'_b', dtype=tf.float32, shape=[output_len],
                            initializer=tf.zeros_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(0.005))
        fc = tf.nn.bias_add(tf.matmul(input, W), b)

        return fc

    def loss_function(self):
        # o1, o2 format: [batch x 1024]
            # compute the L1 difference of two features
        feat_diff = tf.subtract(self.o1, self.o2, name="feat_diff")
        feat_diff_abs = tf.abs(feat_diff, name="feat_diff_abs")
            # FC mapping 1024 -> 2 for binary classification
        fc_output = self.fc_layer(feat_diff_abs, 2, "fc_1024_to_2")
            # convert y_gt to one_hot encoding
        y_gt_onehot = tf.one_hot(self.y_gt, 2)
            # compute 
        loss = tf.losses.softmax_cross_entropy(y_gt_onehot, fc_output)
        return loss

class Siamese_typeC_CE_loss:

    def __init__(self, training=True):
        # train / test switch
        self.is_training = training

        # define input image pair
        self.x1 = tf.placeholder(tf.float32, [None, 128, 128, 3])
        self.x2 = tf.placeholder(tf.float32, [None, 128, 128, 3])

        self.input = tf.keras.layers.concatenate([self.x1, self.x2],
                                                axis=-1,
                                                name="x")

        # define network
        self.output = self.network()

        # define loss
        self.y_gt = tf.placeholder(tf.int32, [None])  # 1 or 0
        self.loss = self.loss_function()

    def network(self):
        conv_1 = tf.layers.conv2d(self.input,
                                  filters=96,
                                  kernel_size=(7,7),
                                  strides=(2,2),
                                  padding='same',
                                  activation=tf.nn.leaky_relu,
                                  name="conv_1")

        mp_1 = tf.layers.max_pooling2d(conv_1,
                                       pool_size=(2,2),
                                       strides=(2,2),
                                       padding='same',
                                       name="mp_1")

        bn_1=tf.layers.batch_normalization(mp_1,
                                           axis=-1,
                                           training=False,
                                           name="bn_1"
                                           )

        conv_2 = tf.layers.conv2d(bn_1,
                                  filters=128,
                                  kernel_size=(5,5),
                                  strides=(1,1),
                                  padding='same',
                                  activation=tf.nn.leaky_relu,
                                  name="conv_2")

        ap_2=tf.layers.average_pooling2d(conv_2,
                                         pool_size=(4,4),
                                         strides=(4,4),
                                         padding = 'same',
                                         name="ap_2")

        mp_2 = tf.layers.max_pooling2d(conv_2,
                                       pool_size=(2,2),
                                       strides=(2,2),
                                       padding='same',
                                       name="mp_2")

        bn_2=tf.layers.batch_normalization(mp_2,
                                           axis=-1,
                                           training=False,
                                           name='bn_2'
                                           )

        conv_3 = tf.layers.conv2d(bn_2,
                                  filters=128,
                                  kernel_size=(5,5),
                                  strides=(1,1),
                                  padding='same',
                                  activation=tf.nn.leaky_relu,
                                  name="conv_3")

        ap_3=tf.layers.average_pooling2d(conv_3,
                                         pool_size=(2,2),
                                         strides=(2,2),
                                         padding = 'same',
                                         name="ap_2")

        mp_3 = tf.layers.max_pooling2d(conv_3,
                                       pool_size=(2,2),
                                       strides=(2,2),
                                       padding='same',
                                       name="mp_3")

        bn_3=tf.layers.batch_normalization(mp_3,
                                           axis=-1,
                                           training=False,
                                           name='bn_3'
                                           )


        conv_4 = tf.layers.conv2d(bn_3,
                                  filters=128,
                                  kernel_size=(3,3),
                                  strides=(1,1),
                                  padding='same',
                                  activation=tf.nn.leaky_relu,
                                  name="conv_4")

        concatenate = tf.keras.layers.concatenate([conv_4,ap_3,ap_2],
                                           axis=-1,
                                           name="concatenate")

        bn_4=tf.layers.batch_normalization(concatenate,
                                           axis=-1,
                                           training=False,
                                           name='bn_4'
                                           )


        conv_5 = tf.layers.conv2d(bn_4,
                                  filters=64,
                                  kernel_size=(1,1),
                                  strides=(1,1),
                                  padding='same',
                                  activation=tf.nn.leaky_relu,
                                  name="conv_5")


        mp_4 = tf.layers.max_pooling2d(conv_5,
                                       pool_size=(2,2),
                                       strides=(2,2),
                                       padding='same',
                                       name="mp_4")

        flatten = tf.layers.flatten(mp_4,
                                    name="flatten")

        fc_1 = tf.layers.dense(flatten, 
                               4096, 
                               activation=tf.nn.leaky_relu,
                               name="fc_1")
        dropout_1 = tf.layers.dropout(fc_1,
                                        rate=0.5,
                                        training=self.is_training,
                                        name="dropout_1")
        fc_2 = tf.layers.dense(dropout_1, 
                               1024, 
                               activation=tf.nn.leaky_relu,
                               name="fc_2")
        dropout_2 = tf.layers.dropout(fc_2,
                                        rate=0.5,
                                        training=self.is_training,
                                        name="dropout_2")
        fc_3 = tf.layers.dense(dropout_2, 
                               512, 
                               activation=tf.nn.leaky_relu,
                               name="fc_3")
        dropout_3 = tf.layers.dropout(fc_3,
                                        rate=0.5,
                                        training=self.is_training,
                                        name="dropout_3")
        fc_4 = tf.layers.dense(dropout_3, 
                               2,
                               name="fc_4")

        return fc_4


    def fc_layer(self, input, output_len, name):
#        print (input.get_shape())
        input_len = input.get_shape()[1]

        # ac = tf.nn.relu(..)
        W = tf.get_variable(name+'_W', dtype=tf.float32, shape=[input_len, output_len], 
                            initializer=tf.keras.initializers.random_normal,
                            regularizer=tf.contrib.layers.l2_regularizer(0.005))
        b = tf.get_variable(name+'_b', dtype=tf.float32, shape=[output_len],
                            initializer=tf.zeros_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(0.005))
        fc = tf.nn.bias_add(tf.matmul(input, W), b)

        return fc

    def loss_function(self):
        # output format: [batch x 2]
        loss = tf.losses.sparse_softmax_cross_entropy(self.y_gt, self.output)
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
# tensorboard --logdir="C:\Users\weixing\Documents\code\Nvidia_AIC_2019\AICity_task2\TF_implementation\logs" --host=127.0.0.1
# tensorboard --logdir="C:\Users\Daniel\Desktop\AICity_task2\TF_implementation\logs" --host=127.0.0.1
        # from tensorflow.keras.applications import mobilenet
        # net = Siamese_classic_mobilenet_CE_loss()
        # tf.global_variables_initializer().run()
        # import numpy as np
        # x = np.ones((6,224,224,3))
        # x2 = np.zeros((6,224,224,3))
        # y = np.array([1,1,1,0,0,0])
        # out = sess.run(net.loss, feed_dict={net.x1:x, net.x2:x2, net.y_gt:y})
        # import pdb; pdb.set_trace() ###
        # print('shit')
        # # print(net(x.astype('float32')).eval())
        # # file_writer = tf.summary.FileWriter('./graph_mn/')
        # # file_writer.add_graph(sess.graph)
        # # file_writer.close()

        import numpy as np
        x1 = np.ones((2,128,128,3))
        x2 = np.zeros((2,128,128,3))
        y = np.array([1,0])
        net = Siamese_typeC_CE_loss(training=False)
        tf.global_variables_initializer().run()
        out = sess.run(net.output, feed_dict={net.x1:x1, net.x2:x2})
        print("out.shape: "+str(out.shape))
        print(out)