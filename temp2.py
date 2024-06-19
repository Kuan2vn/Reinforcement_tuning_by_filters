import numpy as np
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, ZeroPadding2D, BatchNormalization, Activation

# Define a simple model with a convolutional layer
input_layer = Input(shape=(128, 32, 3), name='input_2')
x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(input_layer)
conv_layer = Conv2D(64, (7, 7), strides=(2, 2), name='conv1_conv')(x)
x = BatchNormalization(name='conv1_bn')(conv_layer)
x = Activation('relu', name='conv1_relu')(x)

model = Model(inputs=input_layer, outputs=x)

# Print model summary to verify the layers
model.summary()

# Custom callback to freeze specific filters in a convolutional layer
class FreezeFiltersCallback(Callback):
    def __init__(self, layer_name, filters_to_freeze):
        super(FreezeFiltersCallback, self).__init__()
        self.layer_name = layer_name
        self.filters_to_freeze = filters_to_freeze

    def on_epoch_end(self, epoch, logs=None):
        layer = self.model.get_layer(self.layer_name)
        weights, biases = layer.get_weights()
        
        # Freeze the specified filters by setting their gradients to zero
        for filter_index in self.filters_to_freeze:
            weights[..., filter_index] = np.zeros_like(weights[..., filter_index])
        
        # Set the modified weights back to the layer
        layer.set_weights([weights, biases])

# Instantiate the custom callback
freeze_filters_callback = FreezeFiltersCallback(layer_name='conv1_conv', filters_to_freeze=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# Compile the model (example with dummy loss and optimizer)
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Train the model with the custom callback (example with dummy data)
# model.fit(x_train, y_train, epochs=10, callbacks=[freeze_filters_callback])
