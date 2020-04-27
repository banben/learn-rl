import os
import sys
from game.flappy_bird import GameState
import random
import numpy as np
import cv2

import torch
import torch.optim as optim
import torch.nn.functional as F
from model import DRQN
from memory import Memory
from tensorboardX import SummaryWriter

from config import env_name, initial_exploration, batch_size, update_target, goal_score, log_interval, device, replay_memory_capacity, lr, sequence_length

from collections import deque

def get_action(state, target_net, epsilon, env, hidden):
    action = torch.zeros([2], dtype=torch.float32)

    action_index, hidden = target_net.get_action(state, hidden)
    
    if np.random.rand() <= epsilon:
        print("Performed random action!")
        action_index = [torch.randint(2, torch.Size([]), dtype=torch.int)][0]

    action[action_index] = 1

    return action, hidden

def update_target_model(online_net, target_net):
    # Target <- Net
    target_net.load_state_dict(online_net.state_dict())

def resize_and_bgr2gray(image):
    image = image[0:288, 0:404]
    image_data = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)
    image_data[image_data > 0] = 255
    image_data = np.reshape(image_data, (84, 84, 1))
    return image_data

def image_to_tensor(image):
    image_tensor = image.transpose(2, 0, 1)
    image_tensor = image_tensor.astype(np.float32)
    image_tensor = torch.from_numpy(image_tensor)
    if torch.cuda.is_available():  # put on GPU if CUDA is available
        image_tensor = image_tensor.cuda()
    return image_tensor

def main():
    env = GameState()

    # num_inputs = env.observation_space.shape[0]
    num_inputs = 3136
    num_actions = 2
    print('state size:', num_inputs)
    print('action size:', num_actions)

    online_net = DRQN(num_inputs, num_actions)
    target_net = DRQN(num_inputs, num_actions)
    update_target_model(online_net, target_net)

    optimizer = optim.Adam(online_net.parameters(), lr=lr)
    writer = SummaryWriter('logs')

    online_net.to(device)
    target_net.to(device)
    online_net.train()
    target_net.train()
    memory = Memory(replay_memory_capacity)
    epsilon = 1.0
    loss = 0
    iteration = 0

    while iteration < 2000000:
        done = False

        action = torch.zeros([2], dtype=torch.float32)
        action[0] = 1
        image_data, reward, done = env.frame_step(action)
        image_data = resize_and_bgr2gray(image_data)
        image_data = image_to_tensor(image_data)
        state = image_data
        state = torch.Tensor(state).to(device)

        hidden = None

        while not done:

            action, hidden = get_action(state, target_net, epsilon, env, hidden)
            image_data, reward, done = env.frame_step(action)
            image_data = resize_and_bgr2gray(image_data)
            image_data = image_to_tensor(image_data)

            next_state = image_data
            next_state = torch.Tensor(next_state)

            mask = 0 if done else 1
            reward = reward if not done else -1

            memory.push(state, next_state, action, reward, mask)

            state = next_state

            
            if iteration > initial_exploration and len(memory) > batch_size:
                epsilon -= 0.00005
                epsilon = max(epsilon, 0.1)

                batch = memory.sample(batch_size)
                loss = DRQN.train_model(online_net, target_net, optimizer, batch)

                if iteration % update_target == 0:
                    update_target_model(online_net, target_net)

            iteration += 1

            if iteration % 25000 == 0:
                torch.save(online_net, "pretrained_model/current_model_" + str(iteration) + ".pth")

            print('iteration: {}'.format(iteration))



if __name__=="__main__":
    main()
