from env import *
from helper import *
from agent import *
import torchvision

dataset = 'freesound'
train_epochs = 1
accuracy_weight = 0.72
num_groups_per_layer = 2
num_layers_unfreeze_per_epoch = 2
param = 'mbv2_freesound.pth'

train, val, test = load_freesound_data()

print('BEGIN TRAINING WITH DATASET', dataset)
model = torchvision.models.mobilenet_v2(pretrained=True)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 41)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model.to(device);

to_unfreeze = 0
num_layers = len(get_conv_layers(model))
one_hot_array = layer_groups_to_one_hot(num_layers, num_groups_per_layer, to_unfreeze)
# print(one_hot_array)
# total_params = count_parameters_for_groups(model, one_hot_array, num_groups)
# print(total_params)
# total_params2 = count_parameters(model, num_groups, to_unfreeze)
# print(total_params2)
# unfreeze_groups(model, one_hot_array, num_layers, num_groups)
env = Environment(model, num_groups_per_layer, num_layers_unfreeze_per_epoch, train_epochs, model_default_weight_dir=param, dataset_name=dataset, accuracy_weight=accuracy_weight)

# for i in range(3):
#     env.reset()
#     done = False
#     while not done:
#         print(env.get_state())
#         reward, done, val_acc, train_acc, temp_state = env.action_step(1)
#         print(f'reward {reward}, done {done}, val_acc {val_acc}, train_acc {train_acc}, temp state {temp_state}')

print(num_layers)
print(num_layers_unfreeze_per_epoch)
print(num_layers//num_layers_unfreeze_per_epoch)


# print(len(env.get_state2()))

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1, momentum=0.9)
# unfreeze_everything(model)

# running_loss = 0.0
# correct = 0
# total = 0

# # Lấy một phần tử duy nhất từ DataLoader
# first_batch = next(iter(train))
# first_image = first_batch[0]
# first_label = first_batch[1]

# # first_image = first_image.unsqueeze(0)  # Kích thước: (1, C, H, W)

# first_image = first_image.to(device)
# first_label = first_label.to(device)

# optimizer.zero_grad()
# outputs = model(first_image)
# loss = criterion(outputs, first_label)
# loss.backward()

# unfreeze_groups(model, one_hot_array, num_layers, num_groups_per_layer)
# # unfreeze_groups_shorten(model, num_groups_per_layer, to_unfreeze)
# optimizer.step()
# running_loss += loss.item()

# for images, labels in train:
#     image = images[0]
#     images = images.to(device)
#     labels = labels.to(device)

#     optimizer.zero_grad()
#     outputs = model(images)
#     loss = criterion(outputs, labels)
#     loss.backward()

#     # unfreeze_groups(model, one_hot_array, num_layers, num_groups)
#     optimizer.step()
#     running_loss += loss.item()

#     _, predicted = torch.max(outputs.data, 1)
#     total += labels.size(0)
#     correct += (predicted == labels).sum().item()

# unfreeze_groups_shorten(model, num_groups_per_layer, to_unfreeze)

 
