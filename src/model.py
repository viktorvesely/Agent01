import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers



class Model(tf.keras.Model):
    def __init__(self, actionSpace, observationSpace, hiddenUnits):
        super(Model, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(observationSpace,))
        hidden_layers = []
        for neurons in hiddenUnits:
            hidden_layers.append(
                layers.Dense(neurons, activation='tanh', kernel_initializer='RandomNormal')
                )
        self.output_layer = tf.keras.layers.Dense(
            actionSpace, activation='linear', kernel_initializer='RandomNormal'
            )
        
        self.hidden_layers = hidden_layers

    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output

    