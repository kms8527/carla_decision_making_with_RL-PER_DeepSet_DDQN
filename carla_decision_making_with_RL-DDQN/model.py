import sys

tmp = ['/home/zz/carmaker_ros/ros/ros1_ws/devel/lib/python2.7/dist-packages',
       '/opt/ros/ros1/lib/python2.7/dist-packages', '/opt/ros/kinetic/lib/python2.7/dist-packages',
       '/home/zz/Downloads/CARLA_0.9.6/PythonAPI/carla-0.9.6-py3.5-linux-x86_64.egg',
       '/home/zz/Downloads/CARLA_0.9.6/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg_FILES']
for index in range(0, len(tmp)):
    if tmp[index] in sys.path:
        sys.path.remove(tmp[index])

if not ('/home/zz/Downloads/CARLA_0.9.6/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg_FILES' in sys.path):
    sys.path.append('/home/zz/Downloads/CARLA_0.9.6/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg_FILES')

if not ('/home/zz/Downloads/CARLA_0.9.6/HDMaps' in sys.path):
    sys.path.append('/home/zz/Downloads/CARLA_0.9.6/HDMaps')
if not ('/home/zz/Downloads/CARLA_0.9.6/PythonAPI/carla' in sys.path):
    sys.path.append('/home/zz/Downloads/CARLA_0.9.6/PythonAPI/carla')

import time
import carla
import collections
import torch
from torch.optim import Adam
from torch import nn
import torch.nn.functional as F
import random

class ReplayBuffer:
    def __init__(self):
        self.buffer_limit = 10000
        self.buffer = collections.deque(maxlen = self.buffer_limit)

    def append(self,sample): #sample은 list
        self.buffer.append(sample)

    def size(self):
        return self.buffer

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out = F.relu(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class DQN(nn.Module):
    def __init__(self, inputs_shape, num_actions):
        super(DQN, self).__init__()

        self.inut_shape = inputs_shape
        self.num_actions = num_actions

        self.features = nn.Sequential(

            nn.Conv2d(self.inut_shape, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(

            nn.Linear(64 * 7 * 7, 512),  # nn.Linear(self.features_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def features_size(self):
        return self.features(torch.zeros(1, *self.inut_shape)).view(1, -1).size(1)


class ResNet_DQN(nn.Module):
    def __init__(self, inputs_shape, num_actions):
        super(ResNet_DQN, self).__init__()

        self.inut_shape = inputs_shape
        self.num_actions = num_actions

        self.inchannel = 32
        self.conv1 = nn.Sequential(

            nn.Conv2d(self.inut_shape, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.layer1 = self.make_layer(ResidualBlock, 64,  1, stride=2)
        self.layer2 = self.make_layer(ResidualBlock, 128, 1, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 1, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 1, stride=2)

        self.fc = nn.Sequential(
            # 2 -> (v, a)
            nn.Linear(512+2, 512), # nn.Linear(self.features_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # v = x[:,6,0,0]
        # a = x[:,7,0,0]

        v = torch.reshape(x[:, 6, 0, 0], (-1, 1))
        a = torch.reshape(x[:, 7, 0, 0], (-1, 1))
        x = x[:, :6, :, :]

        # print(v.shape, a.shape, x.shape)
        # print(v, a)

        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 6)
        x = x.view(x.size(0), -1)

        c = torch.cat((x, v), 1)
        # print(c.shape, c[:,512])
        c = torch.cat((c, a), 1)
        # print(c.shape, c[:,513])
        # exit()
        x = self.fc(c)
        return x


class driving_Agent:

    def __init__(self,inputs_shape,num_actions,gray_seg_img,is_training):

        self.inputs_shape = inputs_shape
        self.num_actions = num_actions
        self.is_training = is_training
        self.batch_size = 10
        self.pre_state = None
        self.state = None
        self.action=None
        self.reward=None
        self.done = None
        self.episode_start= time.time()
        self.buffer = ReplayBuffer()
        self.model  = DQN(self.inputs_shape,self.num_actions)
        self.q_value = None
        self.epsilon = 1
        self.epsilon_min = 0.1
        self.decaying = 0.999

    def act(self,state):

        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.decaying
        else:
            self.epsilon = self.epsilon_min

        if random.random() > self.epsilon or not self.is_training:
            state = torch.tensor(state, dtype = torch.float).cuda() #uint8 ->float
            self.q_value = self.model(state)
            action =  self.q_value.max(1)[1].item() #이게 argmax Q가 맞다고?
        else:
            action = random.randrange(self.num_actions)
        return action

    def step(self, action):
        SECONDS_PER_EPISODE = 100000
        '''
        # Simple Action (action number: 3)
        action_test = action -1
        self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=action_test*self.STEER_AMT))
        '''

        # Complex Action (action number: 9)
        if action == 0:  # forward
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0, brake=0.0))
        elif action == 1:  # left
            self.vehicle.apply_control(carla.VehicleControl(throttle=0, steer=-1, brake=0.0))
        elif action == 2:  # right
            self.vehicle.apply_control(carla.VehicleControl(throttle=0, steer=1, brake=0.0))
        elif action == 3:  # forward_left
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1.0, brake=0.0))
        elif action == 4:  # forward_right
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1.0, brake=0.0))
        elif action == 5:  # brake
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=-0.0, brake=1.0))
        elif action == 6:  # brake_left
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=-1.0, brake=1.0))
        elif action == 7:  # brake_right
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=1.0, brake=1.0))
        elif action == 8:  # none
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=-0.0, brake=0.0))

        '''
        if len(self.collision_hist) != 0:
            done = True
            reward = -200 + (time.time()-self.episode_start)*15/SECONDS_PER_EPISODE
        else:
            done = False
            reward = 1
        '''

        done = False
        if len(self.collision_hist) != 0:
            done = True
            reward = -200
        else:
            reward = 1

        if time.time() > self.episode_start + SECONDS_PER_EPISODE:
            done = True

        return reward, done, None

    def state_update(self,state):
        self.pre_state = self.state
        self.state = state

    def one_epoch(self,state):
        self.state_update(state)

        if self.pre_state is not None:
            history = [self.pre_state, self.action, self.reward, self.state, self.done]
            self.buffer.append(history)

        self.action = self.act(self.state) # epsilon-greedy policy
        self.reward,self.done,_=self.step(self.action)

        for  i, [state, action, reward, next_state, done]  in enumerate(random.sample(self.buffer,self.batch_size)):
           state = state.reshape(self.batch_size,1,-1).to(device) #batch_size 1 96 96















