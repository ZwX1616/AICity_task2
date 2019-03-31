import tensorflow as tf
import numpy as np

from network import Siamese_classic_mobilenet

sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))