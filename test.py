from env import *
from tensorflow import keras
import tensorflow as tf


model = keras.models.load_model('Freesound_Audio_ResNet50_v01.h5')
# train_epochs = 2
# accuracy_weight = 0.75
# env = Environment(model, train_epochs, accuracy_weight = accuracy_weight)

# print('state: ',env.get_state())
# print('current freeze idx: ',env.current_freeze_idx)
# print('reward: ',env.get_reward(0.7))
# # env.current_freeze_idx += 4
# env.action_step(1)
# env.freeze_layers()
# print('state: ',env.get_state())
# print('current freeze idx: ',env.current_freeze_idx)
# print('reward: ',env.get_reward(0.7))
# # env.current_freeze_idx = 0
# # env.freeze_layers()
# env.reset()
# print('state: ' ,env.get_state())
# print('DONE')

resnet_layer = model.get_layer('resnet50')
conv4_layer = resnet_layer.get_layer('conv4_block4_3_conv')
weights, biases = conv4_layer.get_weights()
wei = np.array(weights)
print(wei.shape)
wei2 = wei[..., 0]
print(wei2.shape)

# print(resnet_layer.trainable_variables)
# print(sum([tf.keras.backend.count_params(w) for w in resnet_layer.trainable_variables]))
# for temp in resnet_layer.trainable_variables:
#     print(keras.backend.count_params(temp))
# resnet_layer.summary()