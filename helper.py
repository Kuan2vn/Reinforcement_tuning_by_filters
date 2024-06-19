import numpy as np
import torch
import pickle
from collections import defaultdict
from datagen import *
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from IPython import display
import shutil
import os

def load_freesound_data():
    # Đọc danh sách từ tệp
    with open('feature_dataloader.pkl', 'rb') as file:
        loaded_list = pickle.load(file)

    train = loaded_list['train']
    valid = loaded_list['val']
    test = loaded_list['test']
    return train, valid, test

# Function to get all convolutional layers
def get_conv_layers(model):
    conv_layers = []
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            conv_layers.append((name, layer))
    return conv_layers

# Group filters into a specified number of groups
def group_filters(conv_layer, num_groups):
    num_filters = conv_layer.weight.size(0)
    group_size = num_filters // num_groups
    groups = [list(range(i * group_size, (i + 1) * group_size)) for i in range(num_groups)]
    if num_filters % num_groups != 0:
        groups[-1].extend(range(num_groups * group_size, num_filters))
    return groups

# Convert layer groups to one-hot numpy array
def layer_groups_to_one_hot(num_layers, num_groups, num_groups_to_unfreeze):
    total_groups = num_layers * num_groups
    one_hot_array = np.zeros(total_groups, dtype=int)
    if num_groups_to_unfreeze > 0:
        one_hot_array[-num_groups_to_unfreeze:] = 1
    return one_hot_array

# Convert one-hot index to layer and group
def one_hot_index_to_layer_group(index, num_groups):
    layer = index // num_groups
    group = index % num_groups
    return layer, group

# Freeze the entire model except fully connected layers
def unfreeze_everything(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            for param in module.parameters():
                param.requires_grad = True
        else:
            for param in module.parameters():
                param.requires_grad = True

# Unfreeze specific groups of filters based on the one-hot array
def unfreeze_groups(model, one_hot_array, num_layers, num_groups):
    conv_layers = get_conv_layers(model)
    conv_layer_groups = [group_filters(layer[1], num_groups) for layer in conv_layers]

    for index in range(len(one_hot_array)):
        if one_hot_array[index] == 0:
            layer_index, group_index = one_hot_index_to_layer_group(index, num_groups)
            layer_name, layer = conv_layers[layer_index]
            # print(f'freezing {layer_name} group {group_index}')
            for i in conv_layer_groups[layer_index][group_index]:
                layer.weight.grad[i] = 0
                if layer.bias is not None:
                    layer.bias.grad[i] = 0

# Training function with validation and testing
def train_model(model, train_loader, val_loader, num_groups_to_unfreeze, num_layers, num_groups, num_epochs=10, learning_rate=0.01):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    one_hot_array = layer_groups_to_one_hot(num_layers, num_groups, num_groups_to_unfreeze)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, momentum=0.9)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            unfreeze_groups(model, one_hot_array, num_layers, num_groups)
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)

                val_outputs = model(val_images)
                val_loss = criterion(val_outputs, val_labels)
                val_running_loss += val_loss.item()

                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()

        val_loss = val_running_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

    return train_losses, train_accuracies, val_losses, val_accuracies

# Training function with validation and testing
def train_model_batch(model, train_loader, val_loader, num_groups_to_unfreeze, num_layers, num_groups, num_epochs=10, learning_rate=0.01):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    one_hot_array = layer_groups_to_one_hot(num_layers, num_groups, num_groups_to_unfreeze)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, momentum=0.9)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        first_batch = next(iter(train_loader))
        images = first_batch[0]
        labels = first_batch[1]
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        unfreeze_groups(model, one_hot_array, num_layers, num_groups)
        optimizer.step()
        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            first_val_batch = next(iter(val_loader))
            val_images = first_val_batch[0]
            val_labels = first_val_batch[1]
        
            val_images = val_images.to(device)
            val_labels = val_labels.to(device)

            val_outputs = model(val_images)
            val_loss = criterion(val_outputs, val_labels)
            val_running_loss += val_loss.item()

            _, val_predicted = torch.max(val_outputs.data, 1)
            val_total += val_labels.size(0)
            val_correct += (val_predicted == val_labels).sum().item()

        val_loss = val_running_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

    return train_losses, train_accuracies, val_losses, val_accuracies


# Function to evaluate the model on the test set
def evaluate_model(model, test_loader):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.eval()
    test_running_loss = 0.0
    test_correct = 0
    test_total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for test_images, test_labels in test_loader:
            test_images = test_images.to(device)
            test_labels = test_labels.to(device)

            test_outputs = model(test_images)
            test_loss = criterion(test_outputs, test_labels)
            test_running_loss += test_loss.item()

            _, test_predicted = torch.max(test_outputs.data, 1)
            test_total += test_labels.size(0)
            test_correct += (test_predicted == test_labels).sum().item()

    test_loss = test_running_loss / len(test_loader)
    test_accuracy = 100 * test_correct / test_total
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

    return test_loss, test_accuracy

# Count total parameters for selected groups
def count_parameters_for_groups(model, one_hot_array, num_groups_per_layer):
    total_params = 0
    conv_layers = get_conv_layers(model)
    for index in range(len(one_hot_array)):
        if one_hot_array[index] == 1:
            layer_index, group_index = one_hot_index_to_layer_group(index, num_groups_per_layer)
            layer_name, layer = conv_layers[layer_index]
            group_filters_indices = group_filters(layer, num_groups_per_layer)[group_index]
            for i in group_filters_indices:
                total_params += layer.weight[i].numel()
                if layer.bias is not None:
                    total_params += layer.bias[i].numel()
    return total_params

def count_parameters(model, num_groups, num_groups_to_unfreeze):
    num_layers = len(get_conv_layers(model))
    one_hot_array = layer_groups_to_one_hot(num_layers, num_groups, num_groups_to_unfreeze)
    total_params = count_parameters_for_groups(model, one_hot_array, num_groups)
    return total_params

def unfreeze_groups_shorten(model, num_groups, num_groups_to_unfreeze):
    num_layers = len(get_conv_layers(model))
    one_hot_array = layer_groups_to_one_hot(num_layers, num_groups, num_groups_to_unfreeze)
    unfreeze_groups(model, one_hot_array, num_layers, num_groups)



# FOR TRAINING
plt.ion()

def plot_as_2(scores, long_loss): # mean_scores1)

    # display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()

    plt.subplot(2, 1, 1)
    plt.title('Training Reward...')
    plt.xlabel('Number of Games')
    plt.ylabel('Reward')
    plt.plot(scores, label = 'reward')


    # plt.ylim(ymin=-250, ymax=600)
    # plt.legend()
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))

    # plt.subplot(3, 1, 2)
    # plt.title('Training Short Loss...')
    # plt.xlabel('Number of Games')
    # plt.ylabel('Loss')

    # plt.plot(short_loss, label = 'loss')
    # plt.text(len(short_loss)-1, short_loss[-1], str(short_loss[-1]))

    plt.subplot(2, 1, 2)
    plt.title('Validation Accuracy...')
    plt.xlabel('Number of Games')
    plt.ylabel('Accuracy')
    plt.plot(long_loss, label = 'loss')
    plt.text(len(long_loss)-1, long_loss[-1], str(long_loss[-1]))

    plt.tight_layout()

    plt.show(block=False)
    plt.pause(.1)

def plot_as_3(scores, long_loss, mean_scores1):

    # display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()

    plt.subplot(3, 1, 1)
    plt.title('Training Reward...')
    plt.xlabel('Number of Games')
    plt.ylabel('Reward')
    plt.plot(scores, label = 'reward')
    # plt.ylim(ymin=-250, ymax=600)
    # plt.legend()
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))

    plt.subplot(3, 1, 2)
    plt.title('Validation Accuracy...')
    plt.xlabel('Number of Games')
    plt.ylabel('Accuracy')
    plt.plot(long_loss, label = 'acc')
    plt.text(len(long_loss)-1, long_loss[-1], str(long_loss[-1]))

    plt.subplot(3, 1, 3)
    plt.title('Unfreeze all at state...')
    plt.xlabel('Number of Games')
    plt.ylabel('Loss')
    plt.plot(mean_scores1, label = 'no_of_layers')
    plt.text(len(mean_scores1)-1, mean_scores1[-1], str(mean_scores1[-1]))

    plt.tight_layout()

    plt.show(block=False)
    plt.pause(.1)

import json

# plot_reward[-100:], plot_acc[-100:], plot_action[-100:]

def save_lists_to_json(plot_reward, plot_acc, plot_action, file_name):
    """
    Lưu các danh sách vào một tệp JSON.

    Arguments:
    plot_acc: list, danh sách các giá trị accuracy.
    plot_action: list, danh sách các hành động.
    plot_reward: list, danh sách các giá trị reward.
    file_name: str, tên của tệp để lưu dữ liệu.
    """
    data = {
        "Reward history": plot_reward,
        "Accuracy history": plot_acc,
        "Unfreeze all state history": plot_action
    }

    # Mở tệp để ghi
    with open(file_name, 'w') as json_file:
        json.dump(data, json_file)

    print("Dữ liệu đã được lưu vào file:", file_name)

def copy_file_with_number(source_file, destination_folder, new_number):
    # Kiểm tra xem tệp tin gốc tồn tại hay không
    if os.path.exists(source_file):
        # Kiểm tra xem thư mục đích đã tồn tại chưa, nếu chưa thì tạo mới
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        # Tách tên tệp và phần mở rộng
        file_name, file_extension = os.path.splitext(os.path.basename(source_file))

        # Thêm số vào tên tệp
        new_file_name = f"{file_name}_{new_number}{file_extension}"

        # Đường dẫn đầy đủ đến tệp mới
        destination_file = os.path.join(destination_folder, new_file_name)

        # Sao chép tệp tin
        shutil.copy(source_file, destination_file)
        return destination_file
    else:
        return None
    
def save_list_to_txt(lst, filename):
    with open(filename, 'w') as file:
        for row in lst:
            file.write(' '.join(map(str, row)) + '\n')    

def to_dict(game, val_acc, reward, best_reward, best_acc, state):
    state = json.dumps(state)
    game_data = {
        "game": game,
        "val_acc": val_acc,
        "reward": reward,
        "best_reward": best_reward,
        "best_acc": best_acc,
        "epoch_unfreeze": state
    }
    return game_data

def save_json(name, data):
    filename = name + '.json'
    # Lưu dữ liệu vào file JSON
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)