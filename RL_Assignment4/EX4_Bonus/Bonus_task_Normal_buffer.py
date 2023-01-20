# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 18:00:43 2021
Don't chnage the relative position of files.
@author: hongh
"""

import math, random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
import itertools
from collections import deque
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
        # To do: initialize the loss function (Huberloss) , keywords:  torch.nn.SmoothL1Loss , here we set the reduction scheme = 'mean'
        self.criterion = ...
        
        self.replay_buffer = ReplayBuffer(kwargs.get('memory size',1000000), env.observation_space.shape)
        


    def act(self, state):
        # To do: implement epsilon-greedy, where action is the index (just a number) , not a array/list/tensor.
        # Hint: to speed up, use with torch.no_grad():
        return action


    def test_act(self, state):
        # To do: implement greedy policy, where action is the index (just a number) , not a array/list/tensor.
        # Hint: to speed up, use with torch.no_grad():
         
        return action
    
   
    def update(self, batch_size):       
       # Very similar to previous DDQN, except for the loss definition and update manner of target network
        self.q_net1.train()
        
        transitions = self.replay_buffer.sample(batch_size)        
        states, actions, rewards, next_states, dones = transitions
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # To do: Implement DDQN here, should be similar/same as DDQN in previous task.

        # To do:  q loss using self.criterion
        q1_loss = self.criterion(...)       

        # update q networks        
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        # --------To do: Perform Hard update of target networks every self.hard_update_interval-----------     
        ...
        
        self.update_step += 1
        
        # just for monitoring Q loss, in case of wrong implementation, this could diverge to insane values.
        if self.update_step % 1000 == 999:
            print("Q loss: {}".format(q1_loss))


class ReplayBuffer(object):
    
    def __init__(self, max_size, state_dim):
        self.buffer = deque(maxlen=max_size)
        self.state_dim = state_dim
    
    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    
    def sample(self, batch_size):
        # To do: Same from previous task


    def __len__(self):
        return len(self.buffer)


def learn(agent, **kwargs):   
    # To do :You will save the statistics here
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
            # ---IMPORTNT: To do: save the statistics here for plotting ---
            Save the statistics.....
        #----------------------------Testing-----------------------
        if episode%100 == 99:
            test_returns = np.zeros(15)
            print ('-----start Testing------')
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
    test_env = wrap_pytorch(test_env)
    
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
    config.update({'memory size': 1000000}) # 1M frame memory buffer takes around 8GB in RAM. For local computer, you could use 1M memory size. For Colab, size should be <= 0.3M.
    config.update({'target network update interval': 2500}) # every 10000 training steps, update the target model.
    config.update({'max episodes': 1000000}) # you don't need to run that long, but just keep running.
    config.update({'max steps per episode': 27500}) 
    config.update({'batch size': 32}) 
    config.update({'update every n steps': 4})
    config.update({'number of pre-interactions to start training': 10000})
    config.update({'training_env': env})
    config.update({'testing_env': test_env})
    #  agent
    agent = LearningAgent(**config)
    
    # train
    learn(agent, **config)