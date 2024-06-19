from env import *
from helper import *
from agent import *
import torchvision

import time

# DETERMINE PARAMETERS
# train_dir = 'Data/bbc_train.pth'
# val_dir = 'Data/bbc_val.pth'
# model_weight_dir = 'Model_weights/bert_default_weights_bbc.pth'
# no_classify = 5

dataset = 'freesound'
train_epochs = 1            # ALMOST CONSTANT
accuracy_weight = 0.72
num_groups_per_layer = 2
num_layers_unfreeze_per_epoch = 5
param = 'mbv2_freesound.pth'
n_train = 1000


model = torchvision.models.mobilenet_v2(pretrained=True)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 41)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model.to(device);

# bbc_param = ['Data/bbc_train.pth', 'Data/bbc_val.pth', 'Model_weights/bert_default_weights_bbc.pth', 5]
# R8_param = ['Data/R8_train.pth', 'Data/R8_val.pth', 'Model_weights/bert_default_weights_R8.pth', 8]
# R52_param = ['Data/R52_train.pth', 'Data/R52_val.pth', 'Model_weights/bert_default_weights_R52.pth', 52]

print('----------------')
print('BEGIN TRAINING WITH DATASET', dataset)

if __name__ == '__main__':
    env = Environment(model, num_groups_per_layer, num_layers_unfreeze_per_epoch, train_epochs, model_default_weight_dir=param, dataset_name=dataset, accuracy_weight=accuracy_weight)
    N = 3           # move taken before learning
    batch_size = 64
    n_epochs = 8

    output_folder = 'output_figures/figure'
    output_folder = output_folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    actor_source_file = "ppo/actor/actor.pth"
    critic_source_file = "ppo/actor/critic.pth"
    destination_folder = "best_model"
    destination_folder2 = "saved_model"

    agent = Agent(n_actions=2, batch_size=batch_size,
                    n_epochs=n_epochs,
                    input_dims=env.get_state().shape[0], chkpt_dir='ppo')


    # agent.load_models()

    i = 0

    plot_loss = []
    plot_acc = []
    plot_reward = []
    plot_action = []

    avg_score = 0
    avg_acc = 0

    best_score = 40
    avg_score1 = 0
    best_acc = 0
    best_train_acc = 0
    best_reward = 0

    new_number = 0
    save_number = 0
    best_mae = 2

    unfreeze_state = []
    data = []

    step = 0

    # env.window_size -= 25

    while i < n_train:
          start_time = time.time()
          action_history = []

          unfreeze_state_action = []
          env.reset()
          done = False
          ep_reward = 0
          temp_state = 0

          print('Train number ',i+1)
          while not done:
            observation = env.get_state()
            # print(observation)
            action, prob, val = agent.choose_action(observation)
            # action = 1
            reward, done, acc, train_acc, state = env.action_step(action)
            # print('----FINAL ACC-----', acc)

            # Unfreeze all at state
            if np.sum(observation[2:]) >= env.num_layers-num_layers_unfreeze_per_epoch and action == 1:
                temp_state = state
                # print('yes')

            # unfreeze at state
            if action == 1 and len(unfreeze_state_action) < ((env.num_layers*num_groups_per_layer) / num_layers_unfreeze_per_epoch):
                unfreeze_state_action.append(state)

            # print('Agent choose action: ', action)
            if state < 30:
                action_history.append(action)

            # agent.remember(observation, action, prob, val, reward, done)
            ep_reward += reward
            step += 1

          # Learn, clear memory after (choice: after N games, after N steps in game ...)
        #   if step % N == 0 and step != 0:
        #     loss1 = agent.learn()
        
          unfreeze_state.append(unfreeze_state_action)
          plot_action.append(temp_state)
          plot_acc.append(acc)
          plot_reward.append(ep_reward)

          if i % 5 == 0 and i != 0:
              agent.save_models()

              _ = copy_file_with_number(actor_source_file, destination_folder2, save_number)
              _ = copy_file_with_number(critic_source_file, destination_folder2, save_number)
              save_list_to_txt(unfreeze_state, 'unfreeze_state.txt')
              save_number += 1

          avg_score = np.mean(plot_reward[-5:])
          avg_acc = np.mean(plot_acc[-5:])

          plot_as_3(plot_reward[-100:], plot_acc[-100:], plot_action[-100:])
          game_data = to_dict(i, acc, ep_reward, best_reward, best_acc, unfreeze_state_action)
          data.append(game_data)

          save_json('test', data)
        #   save_lists_to_json(plot_reward, plot_acc, plot_action, file_name='history.json')
          
          if i >= 3 and acc > best_acc:
              best_acc = acc
              best_reward = ep_reward
              best_train_acc = train_acc
              _ = copy_file_with_number(actor_source_file, destination_folder, str(new_number) + '_' + str(temp_state))
              _ = copy_file_with_number(critic_source_file, destination_folder, str(new_number) + '_' + str(temp_state))
              new_number += 1

          if i % 10 == 0 and i != 0:
            filename = os.path.join(output_folder, f'figure_{i}.png')
            plt.savefig(filename)

          plt.savefig('temp.png')

          i += 1

          end_time = time.time()

          # print('average move taken/ game: ', mean_move_taken)
          print("last game actions history: ", action_history)
          print("agent's last 5 games average reward: ", avg_score)
          print("last time model acc: ", acc)
          print(f'current best val acc {best_acc} with training acc {best_train_acc} and reward {best_reward}')
          print("agent's last 5 games average accuracy: ", avg_acc)
          print(f"Episode training time: {round(end_time-start_time, 2)} s")
          print('--------------------')