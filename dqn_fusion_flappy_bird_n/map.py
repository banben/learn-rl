import matplotlib.pyplot as plt
from scipy.ndimage import rotate

import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from game.flappy_bird import GameState


def generate_cam(input_image, conv_output):
    """
        Full forward pass
        conv_output is the output of convolutions at specified layer
        model_output is the final output of the model            
    """      
    conv_output, model_output = self.extractor.forward_pass(input_image)
    if target_index is None:
        target_index = np.argmax(model_output.data.numpy())
    # Target for backprop
    one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
    one_hot_output[0][target_index] = 1
    # Zero grads
    self.model.fc.zero_grad()
    # Backward pass with specified target
    model_output.backward(gradient=one_hot_output, retain_graph=True)
    # Get hooked gradients
    guided_gradients = self.extractor.gradients.data.numpy()[0]
    # Get convolution outputs
    target = conv_output.data.numpy()[0]
    # Get weights from gradients
    # Take averages for each gradient
    weights = np.mean(guided_gradients, axis=(1, 2))
    # Create empty numpy array for cam
    cam = np.ones(target.shape[1:], dtype=np.float32)
    # Multiply each weight with its conv output and then, sum
    for i, w in enumerate(weights):
        cam += w * target[i, :, :]
    cam = cv2.resize(cam, (224, 224))
    cam = np.maximum(cam, 0)
    cam = (cam - np.min(cam)) / (np.max(cam) -
                                 np.min(cam))  # Normalize between 0-1
    cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
    return cam




class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.number_of_actions = 2
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.1
        self.number_of_iterations = 2000000
        self.replay_memory_size = 10000
        self.minibatch_size = 32

        self.conv1a = nn.Conv2d(4, 32, 8, 4)
        self.conv1b = nn.Conv2d(1, 32, 8, 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2a = nn.Conv2d(32, 64, 4, 2)
        self.conv2b = nn.Conv2d(32, 64, 4, 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3a = nn.Conv2d(64, 64, 3, 1)
        self.conv3b = nn.Conv2d(64, 64, 3, 1)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(6272, 512)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(512, self.number_of_actions)

    def forward(self, x):
        outa = self.conv1a(x)
        outa = self.relu1(outa)
        outa = self.conv2a(outa)
        outa = self.relu2(outa)
        outa = self.conv3a(outa)
        outa = self.relu3(outa)
        idx = torch.tensor([3])
        if torch.cuda.is_available():
            idx = idx.cuda()
        xb = torch.index_select(x, 1, idx)
        outb = self.conv1b(xb)
        outb = self.relu1(outb)
        outb = self.conv2b(outb)
        outb = self.relu2(outb)
        outb = self.conv3b(outb)
        outb = self.relu3(outb)
        conv_out = outb.clone()
        outa = outa.view(outa.size()[0], -1)
        outb = outb.view(outb.size()[0], -1)
        out = torch.cat((outa, outb), 1)
        out = out.view(out.size()[0], -1)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)

        return out, conv_out


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.uniform(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)


def image_to_tensor(image):
    image_tensor = image.transpose(2, 0, 1)
    image_tensor = image_tensor.astype(np.float32)
    image_tensor = torch.from_numpy(image_tensor)
    if torch.cuda.is_available():  # put on GPU if CUDA is available
        image_tensor = image_tensor.cuda()
    return image_tensor


def resize_and_bgr2gray(image):
    image = image[0:288, 0:404]
    image_data = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)
    image_data[image_data > 0] = 255
    image_data = np.reshape(image_data, (84, 84, 1))
    return image_data

def resize_and_bgr(image):
    image = image[0:288, 0:404]
    return image

def train(model, start):
    # define Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-6)

    # initialize mean squared error loss
    criterion = nn.MSELoss()

    # instantiate game
    game_state = GameState()

    # initialize replay memory
    replay_memory = []

    # initial action is do nothing
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal = game_state.frame_step(action)
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    # initialize epsilon value
    epsilon = model.initial_epsilon
    iteration = 0

    epsilon_decrements = np.linspace(model.initial_epsilon, model.final_epsilon, model.number_of_iterations)

    # main infinite loop
    while iteration < model.number_of_iterations:
        # get output from the neural network
        output = model(state)[0]

        # initialize action
        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action = action.cuda()

        # epsilon greedy exploration
        random_action = random.random() <= epsilon
        if random_action:
            print("Performed random action!")
        action_index = [torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)
                        if random_action
                        else torch.argmax(output)][0]

        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action_index = action_index.cuda()

        action[action_index] = 1

        # get next state and reward
        image_data_1, reward, terminal = game_state.frame_step(action)
        image_data_1 = resize_and_bgr2gray(image_data_1)
        image_data_1 = image_to_tensor(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        action = action.unsqueeze(0)
        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)

        # save transition to replay memory
        replay_memory.append((state, action, reward, state_1, terminal))

        # if replay memory is full, remove the oldest transition
        if len(replay_memory) > model.replay_memory_size:
            replay_memory.pop(0)

        # epsilon annealing
        epsilon = epsilon_decrements[iteration]

        # sample random minibatch
        minibatch = random.sample(replay_memory, min(len(replay_memory), model.minibatch_size))

        # unpack minibatch
        state_batch = torch.cat(tuple(d[0] for d in minibatch))
        action_batch = torch.cat(tuple(d[1] for d in minibatch))
        reward_batch = torch.cat(tuple(d[2] for d in minibatch))
        state_1_batch = torch.cat(tuple(d[3] for d in minibatch))

        if torch.cuda.is_available():  # put on GPU if CUDA is available
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            state_1_batch = state_1_batch.cuda()

        # get output for the next state
        output_1_batch = model(state_1_batch)

        # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
        y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                  else reward_batch[i] + model.gamma * torch.max(output_1_batch[i])
                                  for i in range(len(minibatch))))

        # extract Q-value
        q_value = torch.sum(model(state_batch) * action_batch, dim=1)

        # PyTorch accumulates gradients by default, so they need to be reset in each pass
        optimizer.zero_grad()

        # returns a new Tensor, detached from the current graph, the result will never require gradient
        y_batch = y_batch.detach()

        # calculate loss
        loss = criterion(q_value, y_batch)

        # do backward pass
        loss.backward()
        optimizer.step()

        # set state to be state_1
        state = state_1
        iteration += 1

        if iteration % 25000 == 0:
            torch.save(model, "pretrained_model/current_model_" + str(iteration) + ".pth")

        print("iteration:", iteration, "elapsed time:", time.time() - start, "epsilon:", epsilon, "action:",
              action_index.cpu().detach().numpy(), "reward:", reward.numpy()[0][0], "Q max:",
              np.max(output.cpu().detach().numpy()))


def test(model):
    game_state = GameState()

    # initial action is do nothing
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal = game_state.frame_step(action)
    image_origin = resize_and_bgr(image_data)
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    while True:
        # get output from the neural network
        output, conv_output = model(state)
        heatmap = np.mean(np.mean(conv_output.detach().numpy(), axis=0), axis=0)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        heatmap = cv2.resize(heatmap, (image_origin.shape[1], image_origin.shape[0]))
        heatmap = np.fliplr(rotate(heatmap, 90*3))

        image_origin = np.fliplr(rotate(image_origin, 90*3))
        plt.imshow(image_origin)
        plt.imshow(heatmap, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest', vmin=0, vmax=1)
        plt.show()

        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action = action.cuda()

        # get action
        action_index = torch.argmax(output)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action_index = action_index.cuda()
        action[action_index] = 1

        # get next state
        image_data_1, reward, terminal = game_state.frame_step(action)
        image_origin = resize_and_bgr(image_data_1)
        image_data_1 = resize_and_bgr2gray(image_data_1)
        image_data_1 = image_to_tensor(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        # set state to be state_1
        state = state_1


def main(mode):
    cuda_is_available = torch.cuda.is_available()

    if mode == 'test':
        model = torch.load(
            'pretrained_model/current_model_2000000.pth',
            map_location='cpu' if not cuda_is_available else None
        )

        modelN = NeuralNetwork()
        modelN.load_state_dict(model.state_dict())

        if cuda_is_available:  # put on GPU if CUDA is available
            model = modelN.cuda()

        test(modelN)

    elif mode == 'train':
        if not os.path.exists('pretrained_model/'):
            os.mkdir('pretrained_model/')

        model = NeuralNetwork()

        if cuda_is_available:  # put on GPU if CUDA is available
            model = model.cuda()

        model.apply(init_weights)
        start = time.time()

        train(model, start)


if __name__ == "__main__":
    main(sys.argv[1])
