# -*- coding: utf-8 -*-
"""
Created on Wed May 12 22:46:13 2021
This task requires some efforts and is evlauted as the highest difficulty-level so far. Not really difficult, but a lot of details.
You can take the code from Bonus task 1 to use here.
@author: hongh
"""
import math, random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
import itertools
from common.old_atari_wrappers import wrap_pytorch, make_atari, wrap_deepmind
import matplotlib.pyplot as plt
            

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # To do: using torch.nn.init.kaiming_normal_() to initialize m.weight , choose the proper nonlinearity 
        # m.bias can be intialized to be zero, using torch.nn.init.zeros_()
        ...
    elif classname.find('Linear') != -1:
        # To do: Since we use the leakyReLU with slope of 0.2, you need to choose appropriate nonlinearity slope.
        # bias and weights need to be intialized respectively.
        ...


class QNetwork(nn.Module):    
    def __init__(self, input_shape, num_actions):
        super(QNetwork, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions

        # build your network according to the paper "Playing Atari with Deep Reinforcement Learning"
        # To prevent "dead neurons" of ReLU activation, you could use instead Leakyrelu with the negative slope of 0.2 after the fully-connected layer with 256 neurons.       
        # Hint: nn.sequential() , nn.Conv2d(), nn.Linear()....
        
        # To do: self.features only contains the 3 convolutional blocks and their nonliner activation functions
        self.features = nn.Sequential(
            ....
        )
        # To do: apply the weight initialization scheme on self.features using the defined function init_weights.
        
        # To do: Initialize fully connected layer () , then followed by leaklyReLU with slope 0.2, (We don't use ReLU here), and last linear layer
        self.fc1 = ...
        self.leaky_relu = ...
        self.fc2 = ...
        
        
        
    def forward(self, state):
        # To do: before forwarding the state, the state should be normalized between [0,1], the original pixel range is [0,255]

        return x
        

    def feature_size(self):
        with torch.no_grad():
            return self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)
        
        




class LearningAgent: 
    def __init__(self, **kwargs):      
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using {} for training the algorithm.".format(self.device))        
        self.env = kwargs.get('training_env',None)
        self.gamma = kwargs.get('gamma', 0.99)
        self.hard_update_interval = kwargs.get('target network update interval', 10000)
        
        self.update_step = 0
        
        # ---- initialize networks ----
        self.q_net1 = QNetwork(env.observation_space.shape, env.action_space.n).to(self.device)
        self.target_q_net1 = QNetwork(env.observation_space.shape, env.action_space.n).to(self.device)
                


        # To do: initialize optimizers using Adam optimizer, learning rate is in kwargs.
        self.q1_optimizer = ...
        # To do: initialize the loss function (Huberloss) , keywords:  torch.nn.SmoothL1Loss
        # Note : here the reduction scheme should be set to be 'none', different from previous one. It is needed for incorporating IS-weights.
        self.criterion = ...
        
        # IMPORTANT: In our implementation, we use sum tree, whose root node number must be 2^n. 
        self.replay_buffer = PER(kwargs.get('memory size',1048576), self.env.observation_space.shape)
        


    def act(self, state):
        # To do: implement epsilon-greedy, where action is the index (just a number) , not a array/list/tensor.
        # Hint: to speed up, use with torch.no_grad():
        return action


    def test_act(self, state):
        # To do: implement greedy policy, where action is the index (just a number) , not a array/list/tensor.
        # Hint: to speed up, use with torch.no_grad(): 
        return action
    
   
    def update(self, batch_size):       
        # To do: implement DDQN with prioritized experience replay, now the loss is different due to IS weights.
        self.q_net1.train()
        
        transitions, idxs, weights = self.replay_buffer.sample(batch_size)        
        states, actions, rewards, next_states, dones = transitions
        weights = torch.FloatTensor(weights).to(self.device)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        #DDQN.....
              
        # q loss 
        q1_loss = self.criterion( ... , ...).squeeze() 
          
        # For PER algorithm code line 12,  To do: convert q1_loss from (Cuda) tensor to numpy, 
        td_error_abs = ... + 1e-7 # adding 1e-7 to avoid singulrities.
        # To do: incorporate IS_weight for each sample and then taking the mean.
        q1_loss = #...
        
        
        # update priority for trained samples
        for idx, td_error in zip(idxs, td_error_abs):
            self.replay_buffer.update_priority(idx, td_error)


        # update q networks        
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        # --------To do: Perform Hard update of target networks every self.hard_update_interval-----------     
        ...
        
        self.update_step += 1
        
        # just for monitoring Q
        if self.update_step % 1000 == 999:
            print("Q loss: {}".format(q1_loss))



class SumTree:
    
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.data = np.zeros(capacity, dtype=object) # storing experience
        self.tree = np.zeros(2 * capacity - 1) # storing pripority + parental nodes. If you have n elements in the bottom, then you need n + (n-1) nodes to construct a tree.        
        self.n_entries = 0
        self.overwrite_start_flag = False # record whether N_entry > capacity
        
    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])


    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1 # starting write from the first element of the bottom layer,
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
            self.overwrite_start_flag = True
        if self.n_entries < self.capacity:
            self.n_entries += 1


    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])   
    
    def max_priority(self):
        return np.max(self.tree[self.capacity-1:])

    # ---- modified : to avoid min_prob being overwritten, causing a potential change in max_wi --> Q-loss suddenly changes/rescales --> unstable training  ----
    def min_prob(self):
        ''' To do: 
            return: p_min, the minimal of P(j) in the replay buffer, see line 9 in algorithm in the original paper https://arxiv.org/pdf/1511.05952.pdf.
            p_min is used to determine max w_{i}
        '''
        if self.overwrite_start_flag == False:        
            p_min = ...
            self.p_min_history = p_min
        elif self.overwrite_start_flag == True: 
            p_min = min( ... the same content as p_min before ,self.p_min_history)
            self.p_min_history = min(p_min, self.p_min_history)
        return p_min



class PER(object):
    '''Here we only use  proportional prioritization , not rank-based prioritiation, which is reported to be of better performance.
    '''
    def __init__(self, max_size, shape, alpha=0.6, beta=0.4):
        self.sum_tree = SumTree(max_size)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = (1.0 - beta)/100000.
        self.current_length = 0
        self.frame_stack = shape[0]
        self.state_dim = shape
        
        
    def push(self, state, action, reward, next_state, done):
        # To do: store the priority
        if self.current_length == 0: # The first sample with priority = 1
            priority = 1.0  
        else: # maximal priority
            priority = ...
        self.current_length = self.current_length + 1
        # To Note: we directly priority = td_error ^ self.alpha into the node, which is befneficial for sampling later.
       

        experience = (state, np.array([action]), np.array([reward]), next_state, done)
        self.sum_tree.add(priority, experience)


    def sample(self, batch_size):
        # Perform a Stochastic universal sampling by roulette wheel, which reduces the variance.
        batch_idx, batch, priorities = [], [], []
        segment = self.sum_tree.total() / batch_size
        p_sum = self.sum_tree.tree[0]
    
        for i in range(batch_size): 
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.sum_tree.get(s)
            batch_idx.append(idx)
            batch.append(data)
            priorities.append(p)
            
        # To do: Compute the IS_weights for each sampled experience.           
        prob = ... # a 1-D array with the length of batch size
        IS_weights = ... # a function of prob
        max_weight = ... # Hint : you will need to call self.sum_tree.min_prob()
        IS_weights /= max_weight # should be of the same shape as 'prob'
               
        state_batch = np.empty((batch_size,*self.state_dim))
        action_batch = np.empty((batch_size))
        reward_batch = np.empty((batch_size))
        next_state_batch = np.empty((batch_size,*self.state_dim))
        done_batch = np.empty((batch_size))   

        for i,transition in enumerate(batch):
            state, action, reward, next_state, done = transition
            state_batch[i] = state
            action_batch[i] = action
            reward_batch[i] = reward
            next_state_batch[i] = next_state
            done_batch[i] = done

        # To do: update beta.
        self.beta = ...
        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch), batch_idx, IS_weights

    def update_priority(self, idx, td_error):
        # Slightly different from original code, we save the exponeniated priority as follows to the sum tree, which is easier to take samples according to the probability later.
        # To do: priority = td_error ^ self.alpha
        priority = ...
        self.sum_tree.update(idx, priority)

    def __len__(self):
        return self.current_length


def learn(agent, **kwargs):   
    update_step = 0
    train_start = kwargs.get('number of pre-interactions to start training', 50000)
    batch_size = kwargs.get('batch size', 32)
    try:
        max_steps = env._max_episode_steps# 
    except:
        max_steps = kwargs.get('max steps per episode', 27500)   
    max_episodes = kwargs.get('max episodes', 1000000)   
    update_every_n_step = kwargs.get('update every n steps', 4)   
    n_frame = 0
    running_100_mean = np.zeros(100)    
    epsilon_start = 1
    epsilon_final = 0.1
#    epsilon_decay = 10000
    
    epi_returns = [] # Important: you will save this statistics for plotting

    for episode in range(max_episodes):    
                
        state = env.reset()
        epi_return = 0
        for step in itertools.count():
            # Implement linearly-decaying epsilon
            agent.epsilon = max(epsilon_final, epsilon_start - (epsilon_start - epsilon_final)* n_frame/500000. )
            action= agent.act(state)                            
            next_state, reward, done, _ = env.step(action)            
            n_frame+=1        
            # Push the experience into replay buffer                                                              
            agent.replay_buffer.push(state, action, reward, next_state, done)
            epi_return += reward

            # To do: write an if condition that the training starts after the agent collects enough experience () and performs an update every 4 interactions.
            # Useful variable :  train_start , update_every_n_step , n_frame
            if (len(agent.replay_buffer) > ...) and (...): 
                agent.update(batch_size)
                update_step += 1
            state = next_state
            
            if done == True or step >= max_steps:
                running_100_mean[episode%100] = epi_return
                epi_returns.append(epi_return)
                print('Episode : {} , episodic length : {}, return: {}, exp_count :{}, averged returnover past 100 episodes : {}'.format(episode , step, epi_return, n_frame, np.mean(running_100_mean[:min(100,episode+1)]))) 
                break

        #---------------To Do: Save the satistics of ----------------
        if episode%50 == 49: # you can change here
            epi_returns_np = np.array(epi_returns)
            #To do: save the statistics here for plotting 

        #----------------------------Testing-----------------------
        if episode%100 == 99:
            test_returns = np.zeros(15)
            print ('-------start Testing--------')
            epi_return = 0       
            state = test_env.reset()  
            for i in range(15): # test for 15 episodes
                for t in range(max_steps):
                    # env.render()
                    action = agent.test_act(state)
                    next_state, reward, done, _ = env.step(action)
                    state = next_state
                    epi_return += reward                        
                    if done or (t == max_steps - 1):
                        test_returns[i] = epi_return
                        epi_return = 0
                        break                   
            print('Testing result, Returns: {}'.format(test_returns))


if __name__ == '__main__':  
    
    env_id = "BreakoutNoFrameskip-v4" # you could change to other games, but only use 'NoFrameskip-v4' version. Some games e.g., spaceinvaders require changing the code in wrapper, i.e., frame_skip =3
    env = make_atari(env_id)
    env = wrap_deepmind(env)
    env = wrap_pytorch(env) # Note the env returns the unnormalized pixel [0,255].
    test_env = make_atari(env_id)
    test_env = wrap_deepmind(test_env, episode_life=False, clip_rewards=False)
    test_env = wrap_pytorch(test_env)#wrap_pytorch_with_normalization(env)
    
    # Hint: you could check the shape and content of the state, it is not yet normlized between [0,1]
    obs = env.reset()
    print(obs.shape)
    for i in range(obs.shape[0]):
        plt.imshow(obs[i])
    
    random_seed = None
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)  
        random.seed(random_seed)
        
    # DQN params
    config = {}
    config.update({'gamma': 0.99})
    config.update({'learning rate': 1e-4})
    config.update({'memory size': 262144}) # If you use Prioritized EXP, size must be 2^n. 1M frame memory buffer takes around 8GB in RAM. For local computer, you could use higher memory size.
    config.update({'target network update interval': 3000}) # every 3000 training steps (not interaction steps), update the target model.
    config.update({'max episodes': 1000000}) # you don't need to run that long, but just keep running.
    config.update({'max steps per episode': 27500}) 
    config.update({'batch size': 32}) 
    config.update({'update every n steps': 4})
    config.update({'number of pre-interactions to start training': 50000})
    config.update({'training_env': env})
    config.update({'testing_env': test_env})
    #  agent
    agent = LearningAgent(**config)
    
    # train
    learn(agent, **config)
