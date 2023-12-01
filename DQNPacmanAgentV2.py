import numpy as np
from dataclasses import dataclass
from typing import Any
import torch
from torch import nn
import torch.nn.functional as F
from collections import deque
import random
from collections import namedtuple
import DQNPacmanParametersV2 as dp

originalGameBoard = [
    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
    [3,2,2,2,2,2,2,2,2,2,2,2,2,3,3,2,2,2,2,2,2,2,2,2,2,2,2,3],
    [3,2,3,3,3,3,2,3,3,3,3,3,2,3,3,2,3,3,3,3,3,2,3,3,3,3,2,3],
    [3,6,3,3,3,3,2,3,3,3,3,3,2,3,3,2,3,3,3,3,3,2,3,3,3,3,6,3],
    [3,2,3,3,3,3,2,3,3,3,3,3,2,3,3,2,3,3,3,3,3,2,3,3,3,3,2,3],
    [3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3],
    [3,2,3,3,3,3,2,3,3,2,3,3,3,3,3,3,3,3,2,3,3,2,3,3,3,3,2,3],
    [3,2,3,3,3,3,2,3,3,2,3,3,3,3,3,3,3,3,2,3,3,2,3,3,3,3,2,3],
    [3,2,2,2,2,2,2,3,3,2,2,2,2,3,3,2,2,2,2,3,3,2,2,2,2,2,2,3],
    [3,3,3,3,3,3,2,3,3,3,3,3,1,3,3,1,3,3,3,3,3,2,3,3,3,3,3,3],
    [3,3,3,3,3,3,2,3,3,3,3,3,1,3,3,1,3,3,3,3,3,2,3,3,3,3,3,3],
    [3,3,3,3,3,3,2,3,3,1,1,1,1,1,1,1,1,1,1,3,3,2,3,3,3,3,3,3],
    [3,3,3,3,3,3,2,3,3,1,3,3,3,3,3,3,3,3,1,3,3,2,3,3,3,3,3,3],
    [3,3,3,3,3,3,2,3,3,1,3,4,4,4,4,4,4,3,1,3,3,2,3,3,3,3,3,3],
    [1,1,1,1,1,1,2,1,1,1,3,4,4,4,4,4,4,3,1,1,1,2,1,1,1,1,1,1], # Middle Lane Row: 14
    [3,3,3,3,3,3,2,3,3,1,3,4,4,4,4,4,4,3,1,3,3,2,3,3,3,3,3,3],
    [3,3,3,3,3,3,2,3,3,1,3,3,3,3,3,3,3,3,1,3,3,2,3,3,3,3,3,3],
    [3,3,3,3,3,3,2,3,3,1,1,1,1,1,1,1,1,1,1,3,3,2,3,3,3,3,3,3],
    [3,3,3,3,3,3,2,3,3,1,3,3,3,3,3,3,3,3,1,3,3,2,3,3,3,3,3,3],
    [3,3,3,3,3,3,2,3,3,1,3,3,3,3,3,3,3,3,1,3,3,2,3,3,3,3,3,3],
    [3,2,2,2,2,2,2,2,2,2,2,2,2,3,3,2,2,2,2,2,2,2,2,2,2,2,2,3],
    [3,2,3,3,3,3,2,3,3,3,3,3,2,3,3,2,3,3,3,3,3,2,3,3,3,3,2,3],
    [3,2,3,3,3,3,2,3,3,3,3,3,2,3,3,2,3,3,3,3,3,2,3,3,3,3,2,3],
    [3,6,2,2,3,3,2,2,2,2,2,2,2,1,1,2,2,2,2,2,2,2,3,3,2,2,6,3],
    [3,3,3,2,3,3,2,3,3,2,3,3,3,3,3,3,3,3,2,3,3,2,3,3,2,3,3,3],
    [3,3,3,2,3,3,2,3,3,2,3,3,3,3,3,3,3,3,2,3,3,2,3,3,2,3,3,3],
    [3,2,2,2,2,2,2,3,3,2,2,2,2,3,3,2,2,2,2,3,3,2,2,2,2,2,2,3],
    [3,2,3,3,3,3,3,3,3,3,3,3,2,3,3,2,3,3,3,3,3,3,3,3,3,3,2,3],
    [3,2,3,3,3,3,3,3,3,3,3,3,2,3,3,2,3,3,3,3,3,3,3,3,3,3,2,3],
    [3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3],
    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
]


class DQN(nn.Module):
    def __init__(self,nr,nc,actions,device) -> None: # env is of class Maze
        super(DQN,self).__init__()
        self.device = device
        in_features = int(nr * nc)
        self.layer1 = nn.Linear(in_features,128).to(self.device)
        self.layer2 = nn.Linear(128,64).to(self.device)
        self.layer3 = nn.Linear(64,actions).to(self.device)


    def forward(self,x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    

    def act(self,state):
        state = np.array(state)
        state = state.flatten()
        state_t = torch.as_tensor(state,dtype=torch.float32)
        state_t  = state_t.to(self.device)
        q_values = self(state_t.unsqueeze(0)) # Adding a batch dimension add doing a forward pass
        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()
        return action
    
class CNN2DDQN(nn.Module):
       
    def __init__(self,nd,nr,nc,actions,device) -> None: # env is of class Maze
        super(CNN2DDQN,self).__init__()
        channels = nd
        self.device = device
        print(self.device)
        self.conv1 = nn.Conv2d(in_channels= channels,out_channels= 16,kernel_size=3,padding=1).to(self.device) #default stride is 1
        self.conv2 = nn.Conv2d(in_channels= 16, out_channels= 32,kernel_size=3,padding=1).to(self.device)
        self.layer1 = nn.Linear(nr*nc*32,256).to(self.device)
        self.layer2 = nn.Linear(256,actions).to(self.device)
        self.flatten = nn.Flatten()


    def forward(self,x):
        #print("Input Shape: ", x.shape)
        x = F.relu(self.conv1(x))
        #print("After Conv1: ", x.shape)
        x = F.relu(self.conv2(x))
        #print("After Conv2: ", x.shape)
        x = self.flatten(x)
        #print("After Flatten: ", x.shape)
        x = F.relu(self.layer1(x))
        #print("After Layer1: ", x.shape)
        return self.layer2(x)


    def act(self,state):
        state = np.array(state)
        state_t = torch.as_tensor(state,dtype=torch.float32)
        state_t = state_t.to(self.device)
        q_values = self(state_t.unsqueeze(0)) # Adding a batch dimension add doing a forward pass
        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()
        return action
    

class CNN3DDQN(nn.Module):
       
    def __init__(self,ni,nd,nr,nc,actions,device) -> None: # env is of class Maze
        super(CNN3DDQN,self).__init__()
        channels = ni
        self.device = device
        print(self.device)
        self.conv1 = nn.Conv3d(in_channels= channels,out_channels= 16,kernel_size=3,padding=1).to(self.device) #default stride is 1
        self.conv2 = nn.Conv3d(in_channels= 16, out_channels= 32,kernel_size=3,padding=1).to(self.device)
        self.layer1 = nn.Linear(nd*nr*nc*32,256).to(self.device)
        self.layer2 = nn.Linear(256,actions).to(self.device)
        self.flatten = nn.Flatten()


    def forward(self,x):
        #print("Input Shape: ", x.shape)
        x = F.relu(self.conv1(x))
        #print("After Conv1: ", x.shape)
        x = F.relu(self.conv2(x))
        #print("After Conv2: ", x.shape)
        x = self.flatten(x)
        #print("After Flatten: ", x.shape)
        x = F.relu(self.layer1(x))
        #print("After Layer1: ", x.shape)
        return self.layer2(x)


    def act(self,state):
        state = np.array(state)
        state_t = torch.as_tensor(state,dtype=torch.float32)
        state_t = state_t.to(self.device)
        q_values = self(state_t.unsqueeze(0)) # Adding a batch dimension add doing a forward pass
        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()
        return action
        
class ReplayBuffer(object):
    def __init__(self,DeepQParam:dp.DeepQParams) -> None:
        self.buffer_size = DeepQParam.buffer_size
        self.batch_size = DeepQParam.batch_size
        self.memory = deque([],maxlen=self.buffer_size)
        self.priorities = deque([],maxlen=self.buffer_size)
    

    def push(self, transition:dp.Sarsd):
        self.memory.append(transition)
        self.priorities.append(max(self.priorities,default=1))
    
    def sample_PR(self,priority_scale_val = 1.0):
        
        sample_probs = self.get_probabilities(priority_scale_val)
        sample_indices = random.choices(range(len(self.memory)),k = self.batch_size,weights = sample_probs)
        samples = np.array(self.memory)[sample_indices]
        importance = self.get_importance(sample_probs[sample_indices])
        return samples,importance,sample_indices
    
    def sample(self):
        return random.sample(self.memory,self.batch_size)
    
    def get_probabilities(self,priority_scale):
        #cpu_priorities = self.priorities.cpu().numpy()
        scaled_priorities = np.array(self.priorities) ** priority_scale
        # scaled_priorities = [priority ** priority_scale for priority in self.priorities]
        # total_priority = sum(scaled_priorities)
        sample_probabilities = scaled_priorities/sum(scaled_priorities)
        # sample_probabilities = [priority / total_priority for priority in scaled_priorities]
        # scaled_priorities = [x ** priority_scale for x in self.priorities]
        # sample_probabilities = [x / sum(scaled_priorities) for x in scaled_priorities]
        return sample_probabilities
    
    def get_importance(self,probabilites):
        importance = 1/len(self.memory) * 1/probabilites
        importance_normalized = importance/max(importance)
        return importance_normalized
    
    def set_priorities(self,indices,errors,offset =0.1):
        for i,e in zip(indices,errors):
            self.priorities[i] = abs(e) + offset

    def __len__(self):
        return len(self.memory)



class DQNAgent:
    def __init__(self,charparams:dp.CharacterParams,deepQparams:dp.DeepQParams) -> None:
        self.agent_rep = charparams.agent
        self.space_rep = charparams.space
        self.enemy_rep = charparams.enemy
        self.point_rep = charparams.point
        self.wall_rep = charparams.wall
        self.scare_ghosts = charparams.scared_enemy
        self.pellet_v1 = charparams.pellet_v1
        self.pellet_v2 = charparams.pellet_v2
        self.ghost_safe = charparams.enemy_safe
        self.i = None
        self.j = None
        self.rows = None
        self.columns = None
        self.policy_net = None
        self.target_net = None
        self.rb = None
        self.optimizer = None
        self.batch_size = deepQparams.batch_size
        self.gamma = deepQparams.gamma
        self.lr = deepQparams.optimizer_lr
        self.target_update_hz = deepQparams.target_update_frequency
        self.prev_observation = []
        self.device = None

    def assign_device(self,device):
        self.device = device

    def assign_policy_net(self,policy_net:DQN):
        self.policy_net = policy_net
        self.policy_net = self.policy_net.to(self.device)
    
    
    def get_agent_matrix(self,state):
        agent_matrix = np.where(state == self.agent_rep,1,0)
        #print(agent_matrix)
        return agent_matrix
    
    def get_enemy_matrix(self,state):
        enemy_matrix = np.where(state == self.enemy_rep,1,0)
        #print(enemy_matrix)
        return enemy_matrix
    
    def get_wall_matrix(self,state):
        wall_matrix = np.where(np.isin(state,[self.wall_rep,self.ghost_safe]),1,0)
        #print(wall_matrix)
        return wall_matrix
    
    def get_point_matrix(self,state):
        point_matrix = np.where(state == self.point_rep,1,0)
        #print(point_matrix)
        return point_matrix
    
    def get_pellet_matrix(self, state):
        pellet_matrix = np.where(np.isin(state,[self.pellet_v1,self.pellet_v2]),1,0)
        #print(pellet_matrix)
        return pellet_matrix
    
    def get_scared_ghosts_matrx(self, state):
        scared_ghosts_matrix = np.where(state == self.scare_ghosts,1,0)
        #print(scared_ghosts_matrix)
        return scared_ghosts_matrix
    
    def get_stacked_state_matrix(self,state):
        state = np.array(state)
        agent_matrix = self.get_agent_matrix(state)
        enemy_matrix = self.get_enemy_matrix(state)
        wall_matrix = self.get_wall_matrix(state)
        point_matrix = self.get_point_matrix(state)
        pellet_matrix = self.get_pellet_matrix(state)
        scared_ghost_matrix = self.get_scared_ghosts_matrx(state)
        observation = np.stack((wall_matrix,agent_matrix,point_matrix,pellet_matrix,enemy_matrix,scared_ghost_matrix))
        nd,nr,nc= observation.shape
        return observation

    def get_stacked_state_matrix_MDP(self,state):
        state = np.array(state)
        agent_matrix = self.get_agent_matrix(state)
        enemy_matrix = self.get_enemy_matrix(state)
        wall_matrix = self.get_wall_matrix(state)
        point_matrix = self.get_point_matrix(state)
        pellet_matrix = self.get_pellet_matrix(state)
        scared_ghost_matrix = self.get_scared_ghosts_matrx(state)
        observation = np.stack((wall_matrix,agent_matrix,point_matrix,pellet_matrix,enemy_matrix,scared_ghost_matrix))

        if len(self.prev_observation) == 0 :
            prev_state = observation.copy()
        else:
            prev_state = np.array(self.prev_observation).copy()

        nd,nr,nc= observation.shape

        self.assign_prev_obs(observation.copy())
        observation_stacked = np.stack((observation,prev_state))
        return observation_stacked

    def get_MDP_matrix(self,state):
        state = np.array(state)

        if len(self.prev_observation) == 0:
            prev_state = state.copy()
        else:
            prev_state = np.array(self.prev_observation).copy()
         
        self.assign_prev_obs(state.copy())
        
        MDP_state = np.stack((prev_state,state))
        return MDP_state

    
    def assign_prev_obs(self,observation):
        self.prev_observation = observation



    def assign_target_net(self,target_net:DQN):
        self.target_net = target_net
        self.target_net = self.target_net.to(self.device)
    
    def assign_replay_buffer(self,rb:ReplayBuffer):
        self.rb = rb

    def assign_optimizer(self):
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(),lr=self.lr,amsgrad=True)


    def optimize_model_DQN(self):

        if len(self.rb) < self.batch_size:
            print("Not big enough replay buffer")
            return


        transitions= self.rb.sample()
        state_batch = torch.stack([torch.Tensor(s.state) for s in transitions])
        action_batch = torch.stack([torch.Tensor([s.action]) for s in transitions])
        reward_batch = torch.stack([torch.Tensor([s.reward]) for s in transitions])
        next_state_batch = torch.stack([torch.Tensor(s.next_state) for s in transitions])
        done_batch = torch.stack([torch.Tensor([s.done]) for s in transitions])

        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        done_batch = done_batch.to(self.device)

        state_action_values = self.policy_net(state_batch).gather(1,action_batch.long())
        
        with torch.no_grad():
            self.target_net.eval()
            target_values = self.target_net(next_state_batch)
            self.target_net.train()

        next_state_action_values = target_values.max(dim=1,keepdim=True)[0]

        expected_state_action_values = reward_batch + self.gamma * (1 - done_batch) * next_state_action_values
        
        loss = nn.functional.smooth_l1_loss(state_action_values,expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(),100)
        self.optimizer.step()

        return loss.item()


    def optimize_model_DDQN(self):

        if len(self.rb) < self.batch_size:
            print("Not big enough replay buffer")
            return


        transitions= self.rb.sample()
        state_batch = torch.stack([torch.Tensor(s.state) for s in transitions])
        action_batch = torch.stack([torch.Tensor([s.action]) for s in transitions])
        reward_batch = torch.stack([torch.Tensor([s.reward]) for s in transitions])
        next_state_batch = torch.stack([torch.Tensor(s.next_state) for s in transitions])
        done_batch = torch.stack([torch.Tensor([s.done]) for s in transitions])

        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        done_batch = done_batch.to(self.device)

        state_action_values = self.policy_net(state_batch).gather(1,action_batch.long())
        
        with torch.no_grad():
            self.policy_net.eval()
            self.target_net.eval()
            next_state_actions = self.policy_net(next_state_batch).max(1, keepdim=True)[1]
            next_state_values = self.target_net(next_state_batch).gather(1, next_state_actions.long())
            expected_state_action_values = reward_batch + self.gamma * (1 - done_batch) * next_state_values
            self.policy_net.train()
            self.target_net.train()

        
        loss = nn.functional.smooth_l1_loss(state_action_values,expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(),100)
        self.optimizer.step()

        return loss.item()

    # Failed Prioritise replay implementation
    def optimize_model_DQN_PR(self,epsilon,step):

        if len(self.rb) < self.batch_size:
            print("Not big enough replay buffer")
            return

        priority_scale = 0.0

        if step % 2 == 0:
            priority_scale = 0.7

        transitions,importances,transition_indices= self.rb.sample_PR(priority_scale_val=priority_scale)
        state_batch = torch.stack([torch.Tensor(s.state) for s in transitions])
        action_batch = torch.stack([torch.Tensor([s.action]) for s in transitions])
        reward_batch = torch.stack([torch.Tensor([s.reward]) for s in transitions])
        next_state_batch = torch.stack([torch.Tensor(s.next_state) for s in transitions])
        done_batch = torch.stack([torch.Tensor([s.done]) for s in transitions])

        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        done_batch = done_batch.to(self.device)

        state_action_values = self.policy_net(state_batch).gather(1,action_batch.long())
        
        with torch.no_grad():
            self.target_net.eval()
            target_values = self.target_net(next_state_batch)
            self.target_net.train()

        next_state_action_values = target_values.max(dim=1,keepdim=True)[0]

        expected_state_action_values = reward_batch + self.gamma * (1 - done_batch) * next_state_action_values
        
        errors = torch.subtract(expected_state_action_values,state_action_values)
        errors = [ x.detach().item() for x in errors]

        loss = nn.functional.smooth_l1_loss(state_action_values,expected_state_action_values,reduction='none')
        loss = loss * torch.as_tensor((importances ** (1-epsilon))).to(self.device)
        mean_loss = torch.mean(loss)

        # transition_indices  = torch.as_tensor(transition_indices).to(self.device)
        self.rb.set_priorities(transition_indices,errors)

        self.optimizer.zero_grad()
        mean_loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(),100)
        self.optimizer.step()

        return mean_loss.item()
    

    def get_valid_actions(self,state):
        valid_moves = []
        #print(state)
        self.get_agent_location(state)
        nr,nc = state.shape
        #Check up
        if self.check_in_bounds(self.i-1,self.j,nr,nc):
            valid_moves.append(dp.UP)
        #Check right
        if self.check_in_bounds(self.i,self.j+1,nr,nc):
            valid_moves.append(dp.RIGHT)
        #Check down
        if self.check_in_bounds(self.i+1,self.j,nr,nc):
            valid_moves.append(dp.DOWN)   
        #Check right
        if self.check_in_bounds(self.i,self.j-1,nr,nc):
            valid_moves.append(dp.LEFT)

        #print(valid_moves)
        return valid_moves
    
    def check_in_bounds(self,i,j,nr,nc):
        return i >= 0 and i < nr and j >= 0 and j < nc 

    def pick_random_action(self,state):
        valid_moves = [0,1,2,3]
        random.shuffle(valid_moves)
        action = valid_moves[0]
        return action
    
    def pick_dqn_action(self,state):
        action = self.policy_net.act(state)
        return action
    
    def get_action(self,state):
        action = self.pick_dqn_action(state)
        return action

    def get_agent_location(self,state):
        location = np.where(state == self.agent_rep)
        self.i,self.j = location



def main():
    state = originalGameBoard.copy()
    count = sum(row.count(dp.GOAL) for row in state)
    print(count)

    # character_params = dp.CharacterParams(dp.PATH,dp.GOAL,dp.WALL,dp.GHOST_SAFE_SPACE,dp.SPECIAL_PELLETS_V1,dp.SPECIAL_PELLETS_V2,dp.ENEMY,dp.SCARED_GHOSTS,dp.AGENT)
    # deepQ_params = dp.DeepQParams(dp.GAMMA,dp.BATCH_SIZE,dp.BUFFER_SIZE,dp.MIN_REPLAY_SIZE,dp.TARGET_UPDATE_FREQUENCY,dp.OPTIMIZER_LR)
    # a = DQNAgent(character_params,deepQ_params)
    # np.set_printoptions(threshold=np.inf)
    # # state = np.zeros((4,4))
    # # state[2,2] = character_params.agent
    # # print(state)
    # state = originalGameBoard.copy()
    # state = np.array(state)
    # mdp = a.get_MDP_matrix(state)
    # print(mdp)
    # state[0,0] = 8
    # mdp = a.get_MDP_matrix(state)
    # print(mdp)
    # print(mdp.shape)


    #a.pick_random_action(state)


if __name__ == '__main__':
    main()

