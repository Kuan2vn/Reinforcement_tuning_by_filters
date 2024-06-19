import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import pickle
from helper import *
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torch import cuda

import numpy as np

class Environment:
  def __init__(self, model, num_groups, no_layers_unfreeze_per_epochs, train_epochs, model_default_weight_dir, dataset_name, accuracy_weight = 0.7, total_epochs = 65):
    # DATA = PICKLES FEATURES = (X_TRAIN, Y_TRAIN)
    self.dataset_name = dataset_name

    self.model_default_weight_dir = model_default_weight_dir
    self.accuracy_weight = accuracy_weight
    self.no_layers_unfreeze_per_epochs = no_layers_unfreeze_per_epochs
    self.data, self.val_data, self.test_data = load_freesound_data()
    self.model = model
    self.device = 'cuda:0' if cuda.is_available() else 'cpu'

    #### ADDITIONAL
    self.num_layers = len(get_conv_layers(self.model))
    self.num_groups = num_groups

    # self.last_layer_idx = len(self.model.layers)
    self.current_freeze_idx = 0

    # Temp freeze
    # one_hot_array = layer_groups_to_one_hot(self.num_layers, self.num_groups, 0)
    # unfreeze_groups(self.model, one_hot_array, self.num_layers, self.num_groups)

    # Get trainable params
    self.max_params = count_parameters(self.model, self.num_groups, 10000)
    self.params = self.max_params
    
    # get initial weight
    self.load_default_weights()
    unfreeze_everything(self.model)

    self.count = 0
    self.max_epochs = total_epochs
    self.max_count = 99        # TODO

    self.train_epochs = train_epochs         #TODO CHANGE THIS #5
    self.step = 0
    self.max_step = self.max_epochs // self.train_epochs       # TODO


    ###### CHANGE NEEDED
    # reward or accuracy
    # _, self.accuracy = self.evaluate_model()
    self.accuracy = 0
    self.start_accuracy = self.accuracy
    self.best_accuracy = 80            # TODO
    self.accuracy_threshold = 80

    self.done = False


  def get_reward(self, accuracy):
    self.params = count_parameters(self.model, self.num_groups, self.current_freeze_idx)
    weight_accuracy = self.accuracy_weight
    weight_params = round(1 - self.accuracy_weight, 2)
    # print('weight_accuracy', weight_accuracy)
    # print('weight_params', weight_params)
    
    # Chia tỷ lệ số lượng trainable parameters theo giới hạn tối đa
    scaled_trainable_params = self.params / self.max_params
    scaled_accuracy = accuracy / 100

    # print('------ accuracy: ', accuracy)
    # print('------ acc reward: ', weight_accuracy * scaled_accuracy)
    # print('------ params reward raw: ', weight_params * scaled_trainable_params)
    # print('------ params reward: ', weight_params * (1 - scaled_trainable_params))
    
    # Tính toán reward dựa trên các ràng buộc của accuracy và số lượng trainable parameters
    reward = weight_accuracy * scaled_accuracy + weight_params * (1 - scaled_trainable_params)
    return reward

  def reset(self):
    # self.data = tf.data.Dataset.load(self.train_dir) # reload data
    # self.val_data = tf.data.Dataset.load(self.val_dir)
    self.load_default_weights() # load default weight
    # self.model.set_weights(self.default_weight) # set initial weight
    self.current_freeze_idx = 0
    self.count = 0
    self.accuracy = self.start_accuracy
    self.step = 0
    self.done = False


  # TODO
  def get_state(self):
    step = self.step/self.max_step
    acc = self.accuracy/100
    # Create an array to represent the freeze/unfreeze status of each layer
    freeze_status = [step, acc] # add self.step
    trainable_status = layer_groups_to_one_hot(self.num_layers, self.num_groups, self.current_freeze_idx)

    trainable_status = np.append(freeze_status, trainable_status)

    return trainable_status
  
  def get_state2(self):
    freeze_status = []
    for name, module in self.model.named_modules():
        print(name)
        if isinstance(module, nn.Conv2d):
            for param in module.parameters():
                 if param.requires_grad == True:
                    freeze_status.append(1)  # Unfreeze
                 else:
                    freeze_status.append(0)  # Freeze
            trainable_status = np.array(freeze_status)
    # trainable_status = np.array(freeze_status)

    return trainable_status
  

  def action_step(self, action):
    # done = False
    step_multiply = 1
    end_reward = 0

    if action == 1:
      # unfreeze next layer
      self.current_freeze_idx += self.no_layers_unfreeze_per_epochs
      pass
    elif action == 0:
      # keep freeze
      pass
    else:
      raise Exception("Invalid action")


    # end_state = self.current_freeze_idx <= -8 or self.count >= self.max_count or self.step > self.max_step
    end_state = self.count >= self.max_count or self.step >= self.max_step
    # unfreeze_groups_shorten(self.model, self.num_groups, self.current_freeze_idx)

    ###### TODO CHANGE ACCURACY HERE
    # for epoch in range(self.train_epochs):
      # train_acc = np.random.uniform(0,100)        # TODO
      # print('one normal train ', epoch)
      # pass
    print(f'Epoch [{self.step+1}/{self.max_epochs}], ', end='')
    train_losses, train_accs, val_losses, val_accs = train_model(self.model, self.data, self.val_data, self.current_freeze_idx, self.num_layers, self.num_groups, self.train_epochs)
    # train_losses, train_accs, val_losses, val_accs = train_model_batch(self.model, self.data, self.val_data, self.current_freeze_idx, self.num_layers, self.num_groups, self.train_epochs)

    train_loss = train_losses[-1]
    train_acc = train_accs[-1]
    val_loss = val_losses[-1]
    val_acc = val_accs[-1]
    # val_acc = 90
    # train_acc = 90

    self.step += 1

    # if end_state:
    #   if self.step < self.max_step:
    #     for epoch in range((self.max_step - self.step)*self.train_epochs): # Train till max epochs
    #       # step_multiply = self.max_step - self.step
    #       # train_acc = np.random.uniform(0,100)
    #       # print('TRAINING FOR ...')
    #       # print('step_multiply: ', step_multiply)
    #       train_loss, train_acc = self.train_model()
    #       # print('runnin epochs', epoch)
    #       pass

    # TODO CHANGEEEE
    # temp_accuracy = 80

    # temp_accuracy = np.random.uniform(0,100)
    # temp_accuracy = 0.7
    # print('temp_accuracy: ', temp_accuracy)
    # print('self.accuracy: ', self.accuracy)
    # print('temp_accuracy/self.accuracy_threshold: ', temp_accuracy/100)

    if val_acc < self.accuracy:
      # bad scenario
      self.count += 1
      # reward = -1
    elif val_acc >= self.accuracy:
      # good scenario
      # reward = 1
      self.accuracy = val_acc         # TODO
    else:
      raise Exception("Invalid accuracy scenario")

    # self.accuracy = val_acc

    # val_acc = val_acc/100
    # print('-----ACCURACY: ', val_acc)
    reward = self.get_reward(val_acc)
    # print('-----REWARD: ', reward)


    if val_acc > self.best_accuracy:
      self.save_weights()
      self.best_accuracy = val_acc

    temp_state = self.step

    reward = reward * step_multiply
    # print('----FINAL REWARD: ', reward)
    if end_state:
      reward = reward * 3
      # print('-----END STATE BONUS REWARD: ', end_reward)
      # if val_acc >= self.accuracy_threshold:
      #   reward = 10
      # else:
      #   reward = -10
      self.done = True
      # self.reset()


    return reward, self.done, val_acc, train_acc, temp_state


    # return reward, done, temp_accuracy, train_acc


  def save_weights(self):
    dir = 'Model_weights/best_weights_' + self.dataset_name + '.pth'
    if not os.path.exists('Model_weights'):
        os.makedirs('Model_weights')
    torch.save(self.model.state_dict(), dir)

  def load_best_weights(self):
    dir = 'Model_weights/best_weights_' + self.dataset_name + '.pth'
    self.model.load_state_dict(torch.load(dir))

  def load_default_weights(self):
    self.model.load_state_dict(torch.load(self.model_default_weight_dir))