import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, ZeroPadding2D, BatchNormalization, Activation
from tensorflow.keras.callbacks import Callback

# Define a simple model with a convolutional layer
input_layer = Input(shape=(128, 32, 3), name='input_2')
x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(input_layer)
conv_layer = Conv2D(64, (7, 7), strides=(2, 2), name='conv1_conv')(x)
x = BatchNormalization(name='conv1_bn')(conv_layer)
x = Activation('relu', name='conv1_relu')(x)

model = Model(inputs=input_layer, outputs=x)
model.summary()

# Custom callback to check the gradients of specific filters
class CheckGradientsCallback(Callback):
    def __init__(self, layer_name):
        super(CheckGradientsCallback, self).__init__()
        self.layer_name = layer_name

    def on_epoch_end(self, epoch, logs=None):
        layer = self.model.get_layer(self.layer_name)
        weights = layer.trainable_weights

        # Ensure gradients are being recorded
        with tf.GradientTape() as tape:
            # Perform a forward pass
            y_pred = self.model(self.model.input, training=True)
            # Compute the loss
            loss = self.model.compiled_loss(self.model.targets, y_pred)
        
        # Compute the gradients
        gradients = tape.gradient(loss, weights)
        
        # Check the gradients for each filter
        for i, grad in enumerate(gradients):
            if grad is not None:
                print(f"Gradient for weight {i} in layer {self.layer_name}: {tf.reduce_sum(tf.abs(grad))}")
            else:
                print(f"Gradient for weight {i} in layer {self.layer_name}: None")

# Instantiate the custom callback
check_gradients_callback = CheckGradientsCallback(layer_name='conv1_conv')

# Compile the model (example with dummy loss and optimizer)
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Train the model with the custom callback (example with dummy data)
# Assuming x_train and y_train are your training data and labels
# model.fit(x_train, y_train, epochs=1, callbacks=[check_gradients_callback])
