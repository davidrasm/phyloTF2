"""
Created on Mon Jun 22 16:25:53 2020

@author: david
"""
import tensorflow as tf

class L2Regularizer():
    
    def __init__(self,factor, offset=0.0):
        
        self.factor = factor
        self.offset = offset # offset weights from particular value
        
    def call(self, weights):
        
        return tf.reduce_sum(self.factor * tf.math.square(weights-self.offset))
    