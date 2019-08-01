#Implementation of Deep Deterministic Gradient with Tensor Flow"
# Author: Steven Spielberg Pon Kumar (github.com/stevenpjg)

import gym
from gym.spaces import Box, Discrete
import numpy as np
from ddpg import DDPG
from ou_noise import OUNoise

def make_env(scenario_name, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:        
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    print ("Hello Simple_Spread Env") 
 
    return env 

env = make_env("simple_spread") 
#specify parameters here:
episodes=64
steps=25 
CA_OBS_SPACE = env.n 
CA_ACTION_SPACE = 5 # env.observation_space[0].shape[0] 
SA_ACTION_SPACE = CA_ACTION_SPACE + env.observation_space[0].shape[0] + 1 
CA_ACTION_BOUND = 5.0 
is_batch_norm = False #batch normalization switch


def ca_reset(): 
    return np.random.choice(SA_ACTION_SPACE, CA_OBS_SPACE) 
def ca_step(action): 
	return np.random.choice(SA_ACTION_SPACE, CA_OBS_SPACE), np.random.choice(10), np.random.choice([True, False]), {}   

def main():
    #Randomly initialize critic,actor,target critic, target actor network  and replay buffer   
    agent = DDPG(env, is_batch_norm, CA_OBS_SPACE, CA_ACTION_SPACE, CA_ACTION_BOUND) 
    exploration_noise = OUNoise(CA_ACTION_SPACE)
    counter=0
    reward_per_episode = 0    
    total_reward=0
    num_states = CA_OBS_SPACE 
    num_actions = CA_ACTION_SPACE 

    print "Number of States:", num_states
    print "Number of Actions:", num_actions
    print "Number of Steps per episode:", steps
    #saving reward:
    reward_st = np.array([0])
      
    
    for i in xrange(episodes):
        print "==== Starting episode no:",i,"====","\n"
        # observation = env.reset()
        observation = ca_reset() 
        reward_per_episode = 0
        for t in xrange(steps):
            #rendering environmet (optional)            
            # env.render()
            x = observation
            action = agent.evaluate_actor(np.reshape(x,[1,num_states]))
            noise = exploration_noise.noise()
            action = action[0] + noise #Select action according to current policy and exploration noise
            print "Action at step", t ," :",action,"\n"
            
            # observation,reward,done,info=env.step(action)
            observation,reward,done,info=ca_step(action) 
            print x,observation,action,reward,done 
            #add s_t,s_t+1,action,reward to experience memory
            agent.add_experience(x,observation,action,reward,done)
            #train critic and actor network
            if counter > 64: 
                agent.train()
            reward_per_episode+=reward
            counter+=1
            #check if episode ends:
            if (done or (t == steps-1)):
                print 'EPISODE: ',i,' Steps: ',t,' Total Reward: ',reward_per_episode
                print "Printing reward to file"
                exploration_noise.reset() #reinitializing random noise for action exploration
                reward_st = np.append(reward_st,reward_per_episode)
                np.savetxt('episode_reward.txt',reward_st, newline="\n")
                print '\n\n'
                break
    total_reward+=reward_per_episode            
    print "Average reward per episode {}".format(total_reward / episodes)    


if __name__ == '__main__':
    main()    