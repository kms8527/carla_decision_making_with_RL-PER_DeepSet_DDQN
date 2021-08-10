import time
import torch
from torch.optim import Adam
from torch import nn
import torch.nn.functional as F
import random
from collections import namedtuple
import matplotlib.pyplot as plt
import matplotlib
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import collections
import numpy as np

Transition = namedtuple('Transition', ['state','x_static', 'action',  'a_log_prob', 'reward', 'next_state'])

gamma = 0.99

class ReplayBuffer:
    def __init__(self,batch_size):
        self.buffer_limit = 10000
        self.buffer = collections.deque(maxlen = self.buffer_limit)
        self.priority = []

        self.batch_size = batch_size

    def plot_buffer(self, epoch):
        matplotlib.rcParams['axes.unicode_minus'] = False
        matplotlib.rcParams['font.family'] = "AppleGothic"

        left_val = 0
        str_val = 0
        right_val = 0
        for sample in self.buffer:
            if sample[2] == -1:
                left_val += 1
            elif sample[2] == 0:
                str_val += 1
            else:
                right_val += 1
        # x=["왼쪽", "직진", "오른쪽"]
        x = ["left", "straight", "right"]

        values = [left_val, str_val, right_val]


        f = open("/home/a/version_2_per_deepset/data/safety2_action_each_nums.txt", 'a')
        data = "%d\t " % epoch
        f.write(data)

        for value in values:
            data = "%d\t" % value
            f.write(data)
        data = "\n "
        f.write(data)
        f.close()

    def uniform_make_minibatch(self,batch_size):
        # s0, a, r, s1, done = zip(*random.sample(self.buffer, batch_size))
        # s0, x0_static, a, r, with_final_s1, with_final_x1_static, done = zip(*random.sample(self.buffer, len(self.buffer)))


        s0, x0_static, a, r, with_final_s1 = zip(*random.sample(self.buffer, batch_size))

        # tuple(tensor, tensor, ..) -> tensor([[list], [list], ...]) #
        s0_= torch.cat(s0).cuda()
        x0_static_ = torch.cat(x0_static)
        a_ = torch.tensor(a, dtype=torch.long)
        r_ = torch.tensor(r, dtype=torch.float)


        # arr1 = np.array(s0)
        # return np.concatenate(s0), a, r, np.concatenate(s1), done   #(32, 6, 96, 96)
        return s0_, x0_static_, a_, r_

    def get_priority_experience_batch(self,batch_size):
        p_sum = np.sum(self.priority)
        prob = self.priority / p_sum
        sample_indices = random.choices(range(len(prob)), k = batch_size, weights = prob)
        # importance = (1/prob) * (1/len(self.priority))
        # importance = np.array(importance)[sample_indices]

        # samples = [self.buffer[i] for i in sample_indices]
        # samples = Transition(*zip(*samples))
        # return samples#, importance
        return sample_indices

    def append(self,sample): #sample은 list
        trans = Transition(sample[0],sample[1],sample[2],sample[3],sample[4],sample[5])
        self.buffer.append(trans)

    def size(self):
        return self.buffer

    def clear(self):
        self.buffer.clear()
        self.priority = []

class PPO():

    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 10
    buffer_capacity = 1000

    def __init__(self,num_state,x_static_size,num_action,extra_num,writer):
        super(PPO, self).__init__()
        self.batch_size = 32
        feature_size = 20
        hidden_size = 80
        self.writer = writer
        self.is_training = True
        self.selection_method = 'random'
        self.deepset = DeepSet(num_state,feature_size,x_static_size,hidden_size,extra_num).cuda()
        self.model = Actor(num_state, feature_size, x_static_size, hidden_size, num_action, extra_num).cuda()
        self.critic_net = Critic(feature_size, x_static_size, hidden_size).cuda()
        self.buffer = ReplayBuffer(self.batch_size)
        self.counter = 0
        self.training_step = 0
        self.actor_loss = 0
        self.value_loss = 0
        self.mean_actor_loss = 0
        self.mean_critic_loss = 0
        self.actor_optimizer = torch.optim.Adam(self.model.parameters(), 1e-3)
        self.critic_net_optimizer = torch.optim.Adam(self.critic_net.parameters(), 3e-3)


    def act(self, state,x_static): #selct action
        """
        state:  'torch.LongTensor' type
        """
        state = state.cuda()
        x_static = x_static.cuda()
        # state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            feature_vector = self.deepset(state,x_static)
            action_prob = self.model(feature_vector)
        c = Categorical(action_prob)
        action = c.sample()
        return action.item()-1, action_prob[:, action.item()].item(), action_prob[:].tolist()[0]

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_param(self):
        torch.save(self.model.state_dict(), '../param/net_param/model' + str(time.time())[:10], +'.pkl')
        torch.save(self.critic_net.state_dict(), '../param/net_param/critic_net' + str(time.time())[:10], +'.pkl')

    # def store_transition(self, transition):
    #     self.buffer.append(transition)
    #     self.counter += 1

    def update(self):
        # state, x_static, action, reward = self.buffer.uniform_make_minibatch(self.batch_size)

        # state = torch.tensor([t.state for t in self.buffer], dtype=torch.float)
        # action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1)
        # reward = [t.reward for t in self.buffer]

        # update: don't need next_state
        # reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        # next_state = torch.tensor([t.next_state for t in self.buff er], dtype=torch.float)
        self.mean_actor_loss = 0
        self.mean_critic_loss = 0

        state = torch.tensor([t.state.tolist() for t in self.buffer.buffer])
        x_static = torch.tensor([t.x_static.tolist() for t in self.buffer.buffer])
        action = torch.tensor([t.action for t in self.buffer.buffer], dtype=torch.long).view(-1, 1).cuda()
        reward = [t.reward for t in self.buffer.buffer]
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer.buffer], dtype=torch.float).view(-1, 1).cuda()
        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + gamma * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float).cuda()
        # print("The agent is updateing....")
        for i in range(self.ppo_update_time):
                # index = self.buffer.get_priority_experience_batch(self.batch_size)

            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer.buffer))), self.batch_size, False):
                # if self.training_step % 1000 == 0:
                #     print('I_ep {} ，train {} times'.format(i_ep, self.training_step))
                # with torch.no_grad():
                Gt_index = Gt[index].view(-1, 1)

                # state_input = state[index[0]]
                # x_state_input = x_static[index[0]]
                # for j in index:
                #     if j == index[0]:
                #         continue
                #     state_input = torch.cat([state_input, state[index[j]]]).cuda()
                #     x_state_input = torch.cat([x_state_input, x_static[index[j]]]).cuda()

                state_input = state[index]
                state_input = torch.cat([t for t in state_input]).cuda()
                x_state_input = x_static[index]
                x_state_input = torch.cat([t for t in x_state_input]).cuda()

                feature_vector = self.deepset(state_input, x_state_input)
                V = self.critic_net(feature_vector)
                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!
                action_prob = self.model(feature_vector).gather(1, action[index]+1)  # new policy

                ratio = (action_prob / old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor network
                self.actor_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                self.mean_actor_loss += self.actor_loss
                self.writer.add_scalar('loss/actor_loss', self.actor_loss, global_step=self.training_step)
                # self.writer.add_scalar('loss/actor_loss', actor_loss, global_step=self.training_step)
                self.actor_optimizer.zero_grad()
                self.actor_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # update critic network
                self.value_loss = F.mse_loss(Gt_index, V)
                self.mean_critic_loss += self.value_loss
                self.writer.add_scalar('loss/value_loss', self.value_loss, global_step=self.training_step)
                self.critic_net_optimizer.zero_grad()
                self.value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1

        self.mean_actor_loss /= (np.ceil(len(self.buffer.buffer))*self.ppo_update_time)
        self.mean_critic_loss /= (np.ceil(len(self.buffer.buffer))*self.ppo_update_time)
        self.buffer.clear()


class Actor(nn.Module):
    def __init__(self, input_size, feature_size, x_static_size, hidden_size, output_size, extra_num):
        super(Actor, self).__init__()
        self.extra_num = extra_num
        self.input_size = input_size
        self.static_size = x_static_size
        self.feature_size = feature_size + x_static_size
        # self.batch_size = batch_size

        self.l1 = nn.Linear(self.feature_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()


        # self.softmax = nn.Softmax(dim = 1)


    def forward(self, feature_vector):
        # print(x.size(0))

        # print("before concat :", out.shape)
        # out = torch.cat((feature_vector, x_static), 1)
        # print("after concat :", out.shape)
        out = self.l1(feature_vector)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.fc(out)
        action_prob = F.softmax(out, dim=1)

        # out = self.softmax(out)
        return action_prob

class Critic(nn.Module):
    def __init__(self, feature_size, x_static_size, hidden_size):
        super(Critic, self).__init__()
        self.feature_size = feature_size + x_static_size
        # self.batch_size = batch_size
        self.l1 = nn.Linear(self.feature_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self,feature_vector):
        # print(x.size(0))

        out = self.l1(feature_vector)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.fc(out)
        # out = self.softmax(out)
        return out

class DeepSet(nn.Module):
    # Zaheer et al., NIPS (2017). https://arxiv.org/abs/1703.06114
    def __init__(self, input_size, feature_size, x_static_size, hidden_size, extra_num):
        super(DeepSet, self).__init__()
        self.extra_num = extra_num
        self.input_size = input_size
        self.static_size = x_static_size
        self.phi_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, feature_size)
        )
        self.rho_network = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, feature_size)
        )

    def forward(self, x, x_static):
        # print(x.size(0))

        x = x.view(-1, self.extra_num, self.input_size)
        x_static = x_static.view(-1, self.static_size)
        # feature_points=torch.zeros(self.feature_size-self.static_size).cuda()
        # for index in x:
        #     feature_points+=self.phi_network(indRex)
        feature_points = self.phi_network(x)
        feature_points_sum = torch.sum(feature_points, 1).squeeze(1)
        out = self.rho_network(feature_points_sum)
        # print("before concat :", out.shape)
        out = torch.cat((out, x_static), 1)
        # print("after concat :", out.shape)

        return out