#!/usr/bin/env/ python
import gym
import numpy as np


def replacing_trace(trace, activeTiles, lam, gamma):
    '''To do: replacing trace update rule
    # trace: old trace 
    # activeTiles: current active tile indices
    # lam: lambda
    # return: updated trace'''
    # To do : update the trace here.
    #active = np.isin(range(len(trace)), activeTiles)
    
    trace*=lam*gamma
    trace[activeTiles] = 1
    
    return trace #modified


#%%
class SARSA_lambda_Learner(object):
    def __init__(self, env, traceUpdate):
        self.obs_shape = env.observation_space.shape
        self.obs_high = env.observation_space.high
        self.obs_low = env.observation_space.low
        self.obs_bins_d1 = 20 # Number of bins to Discretize in dim 1
        self.obs_bins_d2 = 15    # Number of bins to Discretize in dim 2
        self.obs_bins = np.array([self.obs_bins_d1,self.obs_bins_d2])
        self.bin_width = (self.obs_high - self.obs_low) / self.obs_bins
        self.action_shape = env.action_space.n
       
        # To do : modify the inital Q-values
        self.Q = np.ones((self.obs_bins[0]+1, self.obs_bins[1]+1, self.action_shape)) * 0
        self.visit_counts = np.zeros((self.obs_bins[0]+1, self.obs_bins[1]+1, self.action_shape))
        self.alpha = 0.05  # Learning rate
        self.gamma = 1  # Discount factor
        self.epsilon = 1
        # To Do: Initialize the trace for each discretized state-action pair
        self.trace = np.zeros((self.obs_bins[0]+1, self.obs_bins[1]+1, self.action_shape)) #modified
        self.traceUpdate = traceUpdate 
        self.lam = 0.95 # lambda, to be modified in other tasks
#%%        
    def discretize(self, obs):
        return tuple(np.round((obs - self.obs_low) / self.bin_width).astype(int))
#%%
    def get_action(self, obs):
        discretized_obs = self.discretize(obs)
        # Epsilon-Greedy action selection
        if np.random.random() > self.epsilon: # Exploit
            return np.argmax(self.Q[discretized_obs])
        else:  # Explore
            return np.random.choice([a for a in range(self.action_shape)])
#%%
    def learn(self, obs, action, reward, done, next_obs, next_action):
        discretized_obs = self.discretize(obs)
        discretized_next_obs = self.discretize(next_obs)
        discretized_sa = discretized_obs + (action,)
        discretized_next_sa = discretized_next_obs + (next_action,)
        
        # To do: Compute td_delta           
        td_delta = reward + self.gamma*(not done)*self.Q[discretized_next_sa[0],
                                                         discretized_next_sa[1],discretized_next_sa[2]]-self.Q[discretized_sa[0],
                                                         discretized_sa[1],discretized_sa[2]]
        # Update the visit counts as statistics for later analysis.
        self.visit_counts[discretized_obs][action] += 1
        # Update the trace
        self.traceUpdate(self.trace, discretized_sa, self.lam, self.gamma)
        
        # To Do: update the q table
        self.Q +=self.alpha*td_delta*self.trace
        
        # TO DO: post-process the trace after the episode ends
        if done == True:            
            self.trace =np.zeros_like(self.trace)
            #...

#%%
def train(agent, env, MAX_NUM_EPISODES):
    best_reward = -float('inf')
    # Record the total number of interactions
    total_interaction_count = 0
    episodic_returns = np.zeros(MAX_NUM_EPISODES)
    for episode in range(MAX_NUM_EPISODES):
        # Decaying epsilon-greedy
        agent.epsilon = 1 - episode/MAX_NUM_EPISODES
        
        done = False
        # Fetch the initial state
        obs = env.reset()
        episodic_return = 0.0 # episodic reward
        action = agent.get_action(obs)
        step_count_train=0
        while not done:            
            next_obs, reward, done, info = env.step(action)
            total_interaction_count+= 1
            next_action = agent.get_action(next_obs)
            agent.learn(obs, action, reward, done, next_obs, next_action)     
            episodic_return += reward
            action = next_action
            obs = next_obs
            step_count_train += 1
            # if step_count_train >= 100000: # Force stopping after 1000 steps to avoid infinite loops
            #     break  
        episodic_returns[episode] = episodic_return
        if episodic_return > best_reward: 
            best_reward = episodic_return
          
        print("Episode#:{} reward:{} best_reward:{} eps:{}".format(episode,
                                     episodic_return, best_reward, agent.epsilon))
    # Display the total number of interactions (as one measure of convergence speed)
    print('Total number of interactions: {}'.format(total_interaction_count))
    
    # To do: Return the trained policy as an 2D-array
    policy = np.argmax(agent.Q, axis=2)#...
    
    return policy, agent.Q.copy(), agent.visit_counts.copy(), episodic_returns

#%%
def test(agent, env, policy):
    done = False
    obs = env.reset()
    episodic_return = 0
    step_count = 0
    while not done:
        env.render()
        action = policy[agent.discretize(obs)]
        next_obs, reward, done, info = env.step(action)
        obs = next_obs
        episodic_return += reward
        step_count += 1
        if step_count >= 1000: # Force stopping after 1000 steps to avoid infinite loops
            break
    return episodic_return


    
#%%
if __name__ == "__main__":
    ''' 
        TO DO : You need to add code for plotting the result and saving the statistics as in the last assignment.
    '''
    # To Do: later modify the MAX_NUM_EPISODES
    MAX_NUM_EPISODES = 2000 
    env = gym.make('MountainCar-v0').env     # Note: the episode only terminates when cars reaches the target, the max episode length is not clipped to 200 steps.
    agent = SARSA_lambda_Learner(env, replacing_trace)
    learned_policy, Q, visit_counts, episodic_returns = train(agent, env, MAX_NUM_EPISODES)
    np.save('SARSA_' + 'RIC_' + str(agent.lam) + '_' + str(MAX_NUM_EPISODES) +'.npy', Q)
    # save episodic returns
    np.save('SARSA_REWARDS_' + 'RIC_' + str(agent.lam) + '_' + str(MAX_NUM_EPISODES) +'.npy',episodic_returns) 
    # after training, test the policy 10 times.
    for _ in range(10):
        reward = test(agent, env, learned_policy)
        print("Test reward: {}".format(reward))
    env.close()