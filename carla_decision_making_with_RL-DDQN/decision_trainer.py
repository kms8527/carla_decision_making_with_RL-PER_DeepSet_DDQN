import sys

# tmp = ['/home/zz/carmaker_ros/ros/ros1_ws/devel/lib/python2.7/dist-packages',
#        '/opt/ros/ros1/lib/python2.7/dist-packages', '/opt/ros/kinetic/lib/python2.7/dist-packages',
#        '/home/zz/Downloads/CARLA_0.9.6/PythonAPI/carla-0.9.6-py3.5-linux-x86_64.egg',
#        '/home/zz/Downloads/CARLA_0.9.6/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg_FILES']
# for index in range(0, len(tmp)):
#     if tmp[index] in sys.path:
#         sys.path.remove(tmp[index])
#
# if not ('/home/zz/Downloads/CARLA_0.9.6/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg_FILES' in sys.path):
#     sys.path.append('/home/zz/Downloads/CARLA_0.9.6/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg_FILES')
#
# if not ('/home/zz/Downloads/CARLA_0.9.6/HDMaps' in sys.path):
#     sys.path.append('/home/zz/Downloads/CARLA_0.9.6/HDMaps')
# if not ('/home/zz/Downloads/CARLA_0.9.6/PythonAPI/carla' in sys.path):
#     sys.path.append('/home/zz/Downloads/CARLA_0.9.6/PythonAPI/carla')

import time
import carla
import collections
import torch
from torch.optim import Adam
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
from collections import namedtuple

Transition = namedtuple(
    'Transition', ('state', 'static', 'action','reward', 'next_state','next_static','done'))

class ReplayBuffer:
    def __init__(self,batch_size):
        self.buffer_limit = 10000
        self.buffer = collections.deque(maxlen = self.buffer_limit)
        self.priority = []
        self.batch_size = batch_size

    def get_priority_experience_batch(self):
        p_sum = np.sum(self.priority)
        prob = self.priority / p_sum
        sample_indices = random.choices(range(len(prob)), k = self.batch_size, weights = prob)
        # importance = (1/prob) * (1/len(self.priority))
        # importance = np.array(importance)[sample_indices]
        samples = [self.buffer[i] for i in sample_indices]
        samples = Transition(*zip(*samples))
        return samples#, importance


    def append(self,sample): #sample은 list
        self.buffer.append(sample)

    def size(self):
        return self.buffer


    def PER_make_minibatch(self,batch_size):
        # s0, a, r, s1, done = zip(*random.sample(self.buffer, batch_size))

        s0, x0_static, a, r, with_final_s1, with_final_x1_static , done = self.get_priority_experience_batch()

        # tuple(tensor, tensor, ..) -> tensor([[list], [list], ...]) #
        s0_= torch.cat(s0).cuda()
        x0_static_ = torch.cat(x0_static)
        a_ = torch.tensor(a, dtype=torch.long)
        r_ = torch.tensor(r, dtype=torch.float)

        non_fianl_s1 = torch.cat([s for s in with_final_s1 if s is not None])
        non_final_x1_static = torch.cat([x for x in with_final_x1_static if x is not None])
        done_ = torch.tensor(done, dtype=torch.float)

        # arr1 = np.array(s0)
        # return np.concatenate(s0), a, r, np.concatenate(s1), done   #(32, 6, 96, 96)
        return s0_, x0_static_, a_, r_, non_fianl_s1, with_final_s1, non_final_x1_static , done_


    def uniform_make_minibatch(self,batch_size):
        # s0, a, r, s1, done = zip(*random.sample(self.buffer, batch_size))
        # s0, x0_static, a, r, with_final_s1, with_final_x1_static, done = zip(*random.sample(self.buffer, len(self.buffer)))


        s0, x0_static, a, r, with_final_s1, with_final_x1_static , done = zip(*random.sample(self.buffer, batch_size))



        # tuple(tensor, tensor, ..) -> tensor([[list], [list], ...]) #
        s0_= torch.cat(s0).cuda()
        x0_static_ = torch.cat(x0_static)
        a_ = torch.tensor(a, dtype=torch.long)
        r_ = torch.tensor(r, dtype=torch.float)

        non_fianl_s1 = torch.cat([s for s in with_final_s1 if s is not None])
        non_final_x1_static = torch.cat([x for x in with_final_x1_static if x is not None])
        done_ = torch.tensor(done, dtype=torch.float)

        # arr1 = np.array(s0)
        # return np.concatenate(s0), a, r, np.concatenate(s1), done   #(32, 6, 96, 96)
        return s0_, x0_static_, a_, r_, non_fianl_s1, with_final_s1, non_final_x1_static , done_

class decision_driving_Agent:

    def __init__(self,inputs_shape,num_actions,is_training,ROI_length,extra_num,controller):
        self.extra_num = extra_num
        self.inputs_shape = inputs_shape
        self.num_actions = num_actions
        self.is_training = is_training
        self.batch_size = 64
        self.selection_method = None
        self.gamma = 0.9999
        self.buffer = ReplayBuffer(self.batch_size)

        self.model  = DeepSet_D3QN(self.inputs_shape,20,3,80,self.num_actions,self.batch_size,self.extra_num).cuda()
        self.target_model = DeepSet_D3QN(self.inputs_shape,20,3,80,self.num_actions,self.batch_size,self.extra_num).cuda()
        self.epsilon = 1

        self.epsilon_min = 0.005
        self.decaying = 0.999
        self.learning_rate =0.001
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss = 9999999999
        self.ROI_length = ROI_length #(meters)
        # self.td_memory = []
        self.td_Error_epsilon = 0.0001
        self.controller = controller
        self.q_value =0

    def update_td_error_memory(self,epoch,alpha = 0.7):
        self.model.eval()
        self.target_model.eval()


        transitions = self.buffer.buffer
        # print("asdfasdf")
        # print(*zip(*transitions))
        # tmp = [list(i) for i in zip(transitions)][0][0]
        batch = Transition(*zip(*transitions))

        #(tensor([]), tensor([])) ... -> tensor([])
        state_batch = torch.cat(batch.state).cuda()
        static_batch = torch.cat(batch.static).cuda()
        action_batch = torch.tensor(batch.action).cuda()
        reward_batch = torch.tensor(batch.reward).cuda()
        # with_final_next_state = torch.cat(batch.next_state)
        # with_final_next_static = torch.cat(batch.next_static)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None]).cuda()
        non_final_x1_static = torch.cat([x for x in batch.next_static if x is not None]).cuda()

        action_batch = action_batch.view(-1,1)
        # 신경망의 출력 Q(s_t, a_t)를 계산
        state_action_values = self.model(state_batch, static_batch).gather(1, action_batch+1).detach()

        # cartpole이 done 상태가 아니고, next_state가 존재하는지 확인하는 인덱스 마스크를 만듬
        # next_q_values= torch.zeros(len(self.buffer.buffer)).cuda()
        non_final_mask = torch.ByteTensor(
            tuple(map(lambda s: s is not None, batch.next_state))).type(torch.BoolTensor)

        # 먼저 전체를 0으로 초기화, 크기는 기억한 transition 갯수만큼
        next_state_value = torch.zeros(len(self.buffer.buffer)).cuda()
        a_m = torch.zeros(len(self.buffer.buffer)).type(torch.LongTensor).cuda()
        # 다음 상태에서 Q값이 최대가 되는 행동 a_m을 Main Q-Network로 계산
        # 마지막에 붙은 [1]로 행동에 해당하는 인덱스를 구함

        # next_q_values =self.model(non_final_next_states,non_final_x1_static).detach()
        a_m[non_final_mask] = self.model(non_final_next_states,non_final_x1_static).detach().max(1)[1]
        non_final_a_m = a_m[non_final_mask]
        # if k==1 for k in batch.action
        #
        # if
        #     lane_change_safety_action_mask = [ if else for in ]
        strait_action_mask = [False if ego_lane % 1 == 0 else index for index, ego_lane in
                              enumerate(non_final_x1_static[0:-1:3])]

        non_final_a_m = [non_final_a_m[k].item() if k != False else 1 for k in strait_action_mask]
        # 다음 상태가 있는 것만을 걸러내고, size 32를 32*1로 변환
        non_final_a_m = torch.tensor(non_final_a_m).cuda()
        a_m_non_final_next_states = non_final_a_m.view(-1, 1)

        # 다음 상태가 있는 것만을 걸러내고, size 32를 32*1로 변환
        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)

        # 다음 상태가 있는 인덱스에 대해 행동 a_m의 Q값을 target Q-Network로 계산
        # detach() 메서드로 값을 꺼내옴
        # squeeze() 메서드로 size[minibatch*1]을 [minibatch]로 변환
        next_state_value[non_final_mask] = self.target_model(
            non_final_next_states,non_final_x1_static).gather(1, a_m_non_final_next_states).detach().squeeze()

        # next_tartget_q[non_final_mask] = self.target_model(non_final_next_states,non_final_x1_static).gather(1, a_m_non_final_next_states).detach().squeeze()


        # TD 오차를 계산
        td_errors = (reward_batch + self.gamma * next_state_value) - \
                    state_action_values.squeeze()
        p = abs(td_errors + self.td_Error_epsilon)**alpha



        # state_action_values는 size[minibatch*1]이므로 squeeze() 메서드로 size[minibatch]로 변환

        # TD 오차 메모리를 업데이트. Tensor를 detach() 메서드로 꺼내와서 NumPy 변수로 변환하고 다시 파이썬 리스트로 변환
        self.buffer.priority = p.tolist()

    def memorize_td_error(self,td_error):
        '''TD 오차를 메모리에 저장'''

        if len(self.buffer.priority) < self.buffer.buffer_limit:
            self.buffer.priority.append(td_error)  # 메모리가 가득차지 않은 경우


    def __len__(self):
        '''len 함수로 현재 저장된 갯수를 반환'''
        return len(self.buffer.priority)

    def search_extra_in_ROI(self,extra_lists,player,extra_dl_list):
        new_extra_lists = []
        new_extra_dl_lists = []
        ROI_distance = 1000 #(m)
        for n,extra in enumerate(extra_lists):
            distance_between_vehicles = ((extra.get_location().x-player.get_location().x)**2+(extra.get_location().y-player.get_location().y)**2+(extra.get_location().z-player.get_location().z)**2)**0.5
            if distance_between_vehicles < ROI_distance:
                new_extra_lists.append(extra)
                new_extra_dl_lists.append(extra_dl_list[n])
        out = [new_extra_lists, new_extra_dl_lists]
        return out

    def act(self,state,x_static):

        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.decaying
        else:
            self.epsilon = max(self.epsilon,self.epsilon_min)

        # x_static = state[-1]
        # state = state[:-1]

        if random.random() > self.epsilon or not self.is_training:
            self.selection_method = 'max'
            # state= torch.cat(state)
            # x_static = torch.cat(x_static)l
            state = state.type(torch.FloatTensor).cuda() #uint8 ->float
            x_static = x_static.type(torch.FloatTensor).cuda()
            with torch.no_grad():
                self.q_value = self.model(state,x_static)
                action =  int(self.q_value.argmax().item())-1

            print("Q: ",self.q_value, "q_ACTION:",action)

            # print(self.q_value)



        else:
            self.selection_method = 'random'
            action = random.randrange(self.num_actions)-1
            print("random action :", action)
        return action

    def learning(self):  # sample에서 뽑고 backpropagation 하는 과정
        """
        DQN learning
        """
        s0, x0_static, a, r, non_final_s1, with_fianl_s1, non_final_x1_static, done = self.buffer.uniform_make_minibatch(
            self.batch_size)
        # print(s0)

        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, with_fianl_s1))).type(torch.bool)
        next_q_values = torch.zeros(self.batch_size)

        # s0 = torch.tensor(s0, dtype=torch.float)
        # x0_static = torch.tensor(x0_static, dtype= torch.float)
        # non_final_s1 = torch.tensor(non_final_s1, dtype=torch.float)
        # non_final_x1_static = torch.tensor(non_final_x1_static, dtype = torch.float)
        # a = torch.tensor(a, dtype=torch.long)
        # r = torch.tensor(r, dtype=torch.float)
        # done = torch.tensor(done, dtype=torch.float)

        s0 = s0.cuda()
        x0_static = x0_static.cuda()
        non_final_s1 = non_final_s1.cuda()
        non_final_x1_static = non_final_x1_static.cuda()
        a = a.cuda()
        r = r.cuda()
        done = done.cuda()

        ##  forward  ##
        # print("start forward")
        self.model.eval()

        next_q_value = torch.zeros(self.batch_size).type(torch.float).cuda()

        q_values = self.model(s0, x0_static).cuda()
        # action index form main_q
        next_q_value[non_final_mask] = self.model(non_final_s1, non_final_x1_static).detach().max(1)[0].cuda()

        q_value = q_values.gather(1, a.unsqueeze(1) + 1).squeeze(1)
        # print(next_q_values)
        # next_q_value = next_q_values.gather(1, next_q_values.max(1)[1].unsqueeze(1)).squeeze(1)
        expected_q_values = r + self.gamma * next_q_value
        # print("expected_q_values :" , expected_q_values[0], "q_value :", q_value[0])
        # print("finished forward")

        # for x in non_final_mask:
        #     if x == False:
        #         print("here")

        self.model.train()
        self.optimizer.zero_grad()

        # 0 : 좌회전 , 1 : 직진 : 2 : 우회전 시 Q value
        # self.loss = (q_value - expected_q_values.detach()).pow(2).mean()
        self.loss = F.smooth_l1_loss(q_value, expected_q_values)
        # print(" loss : ", self.loss, "q_value :", q_value.mean())
        # next_q_state_values = self.target_model(s1).cuda()

        # off-policy

        # zero the gradients after updating
        ##  backward  ##
        self.loss.backward()

        ##  update weights  ##
        self.optimizer.step()

    def ddqn_learning(self):  # sample에서 뽑고 backpropagation 하는 과정
        """
        DQN learning
        """

        s0, x0_static, a, r, non_final_s1, with_fianl_s1, non_final_x1_static, done = self.buffer.PER_make_minibatch(
            self.batch_size)
        # print("finished pick minibatch data")
        # print(s0)



        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, with_fianl_s1))).type(torch.bool)
        # next_q_values = torch.zeros(self.batch_size)

        # s0 = torch.tensor(s0, dtype=torch.float)
        # x0_static = torch.tensor(x0_static, dtype= torch.float)
        # non_final_s1 = torch.tensor(non_final_s1, dtype=torch.float)
        # non_final_x1_static = torch.tensor(non_final_x1_static, dtype = torch.float)
        # a = torch.tensor(a, dtype=torch.long)
        # r = torch.tensor(r, dtype=torch.float)
        # done = torch.tensor(done, dtype=torch.float)

        s0 = s0.cuda()
        x0_static = x0_static.cuda()
        non_final_s1 = non_final_s1.cuda()
        non_final_x1_static = non_final_x1_static.cuda()
        a = a.cuda()
        r = r.cuda()
        done = done.cuda()

        ##  forward  ##
        # print("start forward")
        self.model.eval()
        self.target_model.eval()

        next_tartget_q = torch.zeros(self.batch_size).cuda()
        a_m = torch.zeros(self.batch_size).type(torch.LongTensor).cuda()

        q_values = self.model(s0, x0_static).cuda()
        # action index form main_q

        a_m[non_final_mask] = self.model(non_final_s1, non_final_x1_static).detach().max(1)[1].cuda()
        non_final_a_m = a_m[non_final_mask]
        strait_action_mask = [False if ego_lane % 1 == 0 else index for index, ego_lane in enumerate(non_final_x1_static[0:-1:3])]
        non_final_a_m = [non_final_a_m[k].item() if k != False else 1 for k in strait_action_mask]
        # 다음 상태가 있는 것만을 걸러내고, size 64를 64*1로 변환
        non_final_a_m = torch.tensor(non_final_a_m).cuda()
        a_m_non_final_next_states = non_final_a_m.view(-1, 1)

        # 다음 상태가 있는 인덱스에 대해 행동 a_m의 Q값을 target Q-Network로 계산
        # detach() 메서드로 값을 꺼내옴
        # squeeze() 메서드로 size[minibatch*1]을 [minibatch]로 변환
        next_tartget_q[non_final_mask] = self.target_model(non_final_s1,non_final_x1_static).gather(1, a_m_non_final_next_states).detach().squeeze()
        q_value = q_values.gather(1, a.unsqueeze(1) + 1).squeeze(1)
        expected_state_action_values = r + self.gamma * next_tartget_q
        self.model.train()
        # 0 : 좌회전 , 1 : 직진 : 2 : 우회전 시 Q value
        # self.loss = F.smooth_l1_loss(q_value, expected_state_action_values)
        self.loss = F.mse_loss(q_value, expected_state_action_values).mean()

        # zero the gradients after updating
        self.optimizer.zero_grad()

        ##  backward  ##

        self.loss.backward()

        ##  update weights  ##
        self.optimizer.step()

        self.model.eval()
        self.target_model.eval()

        q_value = self.model(s0, x0_static).cuda().gather(1, a.unsqueeze(1) + 1).squeeze(1)
        next_tartget_q[non_final_mask] = self.target_model(non_final_s1,non_final_x1_static).gather(1, a_m_non_final_next_states).detach().squeeze()
        # expected_state_action_values = r + self.gamma * next_tartget_q
        if self.loss-F.mse_loss(q_value, expected_state_action_values).mean() >0:
            pass
        else:
            print("increase loss")
            self.learning_rate /=2
            self.learning_rate = max(self.learning_rate,0.00001)

# class phi_network(nn.Module):
#     def __init__(self,input_size, hidden_size,feature_size):
#         super(phi_network,self).__init__()
#         self.input_size = input_size
#         self.l1 = nn.Linear(input_size,hidden_size)
#         self.l2 = nn.Linear(hidden_size,hidden_size)
#         self.l3 = nn.Linear(hidden_size,feature_size)
#         self.relue = nn.ReLU()
#     def forward(self,x):
#         x.view(-1)

class DQN(nn.Module):
    def __init__(self,input_size,feature_size,x_static_size, hidden_size, output_size,batch_size,extra_num):
        super(DQN,self).__init__()
        self.extra_num = extra_num
        self.input_size = input_size
        self.static_size = x_static_size
        self.feature_size = feature_size+x_static_size
        self.batch_size = batch_size
        self.phi_network = nn.Sequential(
            nn.Linear(input_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,feature_size),
            # nn.Sigmoid()

        )
        self.rho_network = nn.Sequential(
            nn.Linear(feature_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,feature_size),
            # nn.Sigmoid()

        )
        self.l1 = nn.Linear(self.feature_size,hidden_size)
        self.l2 = nn.Linear(hidden_size,hidden_size)
        self.l3 = nn.Linear(hidden_size,output_size)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim = 1)

    def forward(self,x,x_static):
        # print(x.size(0))

        x=x.view(-1,self.extra_num, self.input_size)
        x_static= x_static.view(-1, self.static_size)
        # feature_points=torch.zeros(self.feature_size-self.static_size).cuda()
        # for index in x:
        #     feature_points+=self.phi_network(index)
        feature_points = self.phi_network(x)
        feature_points_sum = torch.sum(feature_points,1).squeeze(1)
        out = self.rho_network(feature_points_sum)
        # print("before concat :", out.shape)
        out = torch.cat((out,x_static),1)
        # print("after concat :", out.shape)
        out = self.l1(out)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # out = self.softmax(out)
        return out

class DeepSet_D3QN(nn.Module):
    def __init__(self, input_size, feature_size, x_static_size, hidden_size, output_size, batch_size, extra_num):
        super(DeepSet_D3QN, self).__init__()
        self.extra_num = extra_num
        self.input_size = input_size
        self.static_size = x_static_size
        self.feature_size = feature_size + x_static_size
        self.batch_size = batch_size
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
        self.l1 = nn.Linear(self.feature_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.fc3_adv = nn.Linear(hidden_size, output_size)
        self.fc3_v = nn.Linear(hidden_size,1)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim = 1)

    def forward(self, x, x_static):
        # print(x.size(0))

        x = x.view(-1, self.extra_num, self.input_size)
        x_static = x_static.view(-1, self.static_size)
        # feature_points=torch.zeros(self.feature_size-self.static_size).cuda()
        # for index in x:
        #     feature_points+=self.phi_network(index)
        feature_points = self.phi_network(x)
        feature_points_sum = torch.sum(feature_points, 1).squeeze(1)
        out = self.rho_network(feature_points_sum)
        # print("before concat :", out.shape)
        out = torch.cat((out, x_static), 1)
        # print("after concat :", out.shape)
        out = self.l1(out)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        adv = self.fc3_adv(out)
        val = self.fc3_v(out).expand(-1,adv.size(1))
        out = val + adv - adv.mean(1,keepdim=True).expand(-1,adv.size(1))
        # out = self.softmax(out)
        return out


