import torch
import torch.nn as nn
import torch.nn.functional as F

from config import gamma, device, batch_size, sequence_length, burn_in_length
import pdb

class DRQN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DRQN, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.conv1 = nn.Conv2d(1, 32, 8, 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc0 = nn.Linear(num_inputs, 512)

        self.lstm = nn.LSTM(input_size=512, hidden_size=512, batch_first=True)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, num_outputs)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, x, hidden=None):
        # x [batch_size, sequence_length, num_inputs]

        _batch_size = x.size()[0]
        _sequence_length = x.size()[1]

        out = x.view(-1, x.size()[2], x.size()[3], x.size()[4])

        out = self.conv1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = out.view(_batch_size, _sequence_length, self.num_inputs)
        out = self.fc0(out)

        if hidden is not None:
            out, hidden = self.lstm(out, hidden)
        else:
            out, hidden = self.lstm(out)
        out = F.relu(self.fc1(out))
        qvalue = self.fc2(out)

        return qvalue, hidden


    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch):
        def slice_burn_in(item):
            return item[:, burn_in_length:, :]

        states = torch.stack(batch.state)
        next_states = torch.stack(batch.next_state)
        actions = torch.stack(batch.action).view(batch_size, sequence_length, -1).long()
        rewards = torch.stack(batch.reward).view(batch_size, sequence_length, -1)
        masks = torch.stack(batch.mask).view(batch_size, sequence_length, -1)
        if torch.cuda.is_available():
            states = states.cuda()
            next_states = next_states.cuda()
            actions = actions.cuda()
            rewards = rewards.cuda()
            masks = masks.cuda()

        pred, _ = online_net(states)
        next_pred, _ = target_net(next_states)

        pred = slice_burn_in(pred)
        next_pred = slice_burn_in(next_pred)
        actions = slice_burn_in(actions)
        rewards = slice_burn_in(rewards)
        masks = slice_burn_in(masks)
        
        pred = pred.gather(2, actions)
        
        target = rewards + masks * gamma * next_pred.max(2, keepdim=True)[0]

        loss = F.mse_loss(pred, target.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def get_action(self, state, hidden):
        state = state.unsqueeze(0).unsqueeze(0)

        qvalue, hidden = self.forward(state, hidden)

        _, action = torch.max(qvalue, 2)
        
        return action[0][0], hidden
