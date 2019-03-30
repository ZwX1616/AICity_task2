from tensorflow.keras import Input
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.layers import Lambda, Dense
from tensorflow.keras.models import Model
import tensorflow as tf
def get_siamese_model(input_shape):
    """
        Model architecture
    """
    
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    
    # Convolutional Neural Network
    model = inception_v3.InceptionV3()
    
    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    
    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors:abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])
    
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1,activation='sigmoid')(L1_distance)
    
    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    
    # return the model
    return siamese_net

if __name__ == '__main__':
    net = get_siamese_model((299,299,3))
    import pdb; pdb.set_trace()
    with tf.Session() as sess:
        import numpy as np
        x = np.random.normal(0,1,(1,299,299,3))
        sess.run(tf.global_variables_initializer())
        sess.run(net.output, feed_dict = {'input_0': x, 'input_1': x})
        file_writer = tf.summary.FileWriter('./ref2/')
        file_writer.add_graph(sess.graph)
        file_writer.close()