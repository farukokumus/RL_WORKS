# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 19:13:02 2020

@author: hongh
"""

import numpy as np
import gym
import matplotlib.pyplot as plt
import seaborn as sns

env = gym.make('FrozenLake-v0')




# %%
def plot(V):
    %matplotlib inline
    plt.figure(figsize=(8, 8))
    sns.heatmap(V.reshape(4, 4),  cmap="YlGnBu", annot=True, cbar=False, square=True);


# %%
def state_reward(env, V, s, a, gamma):
    
    sum = 0
    for p, sn, r, _ in env.model[s][a]:
        sum += p * (r + gamma * V[sn])
    return sum


# %%
def finding_delta(env, gamma, theta,V):
    
    while True:
        delta = 0
        for s in range(env.nb_states):
            v = V[s]
            V[s] = max([state_reward(env, V, s, a, gamma) for a in range(env.nb_actions)])            
            delta = max(delta, abs(v - V[s]))
    
        if delta < theta: 
            break
    return V

# %%
def output_deterministic(env, V,pi,gamma):
    for s in range(env.nb_states):
        pi[s] = np.argmax([state_reward(env, V, s, a, gamma) for a in range(env.nb_actions)])   
    return pi
# %%

def value_iter(env, gamma, theta):
    """To Do : Implement Policy Iteration Algorithm
    gamma (float) - discount factor
    theta (float) - termination condition
    env - environment with following required memebers:
    
    Useful variables/functions:
        
            env.nb_states - number of states
            env.nb_action - number of actions
            env.model     - prob-transitions and rewards for all states and actions, you can play around that
        
        
        return the value function V and policy pi, 
        pi should be a determinstic policy and an illustration of randomly initialized policy is below
    """
    V = np.zeros(env.nb_states) 
    pi = np.zeros(env.nb_states, dtype=int)
    V=finding_delta(env, gamma, theta,V)
    pi=output_deterministic(env,V,pi,gamma)
    
     
    return V, pi
# %%
if __name__ == '__main__':
    env.reset()
    env.render()

    
    if not hasattr(env, 'nb_states'):  
        env.nb_states = env.env.nS
    if not hasattr(env, 'nb_actions'): 
        env.nb_actions = env.env.nA
    if not hasattr(env, 'model'):      
        env.model = env.env.P
        
    # Check #state, #actions and transition model
    # env.model[state][action]
    #print(env.nb_states, env.nb_actions, env.model[14][2])     
        
  
    V, pi = value_iter(env, gamma=1.0, theta=1e-4)
    print(V.reshape([4, -1]))
    
    
    a2w = {0:'<', 1:'v', 2:'>', 3:'^'}
    policy_arrows = np.array([a2w[x] for x in pi])
    print(np.array(policy_arrows).reshape([-1, 4]))
    
    plot(V)