import gym
from gym.spaces import Box, Discrete
import numpy as np
import random

from ddpg.ddpg import DDPG 
from ddpg.ou_noise import OUNoise 

from dqn.dqn import DQNSolver 
from dqn.scores.score_logger import ScoreLogger 

from config import * 

def ca_reset(): 
	return np.random.choice(SA_ACTION_SPACE, CA_OBS_SPACE) 

""" 
ca_step() is just for testing purposes 
""" 
def ca_step(action): 
	return np.random.choice(SA_ACTION_SPACE, CA_OBS_SPACE), np.random.choice(10), np.random.choice([True, False]), {} 

""" 
sa_state(): To merge the two states received by the individual agents 
(one from central agent and one from the environment) into one vector 
""" 
def sa_state(x, obs, i): 
	one = x
	two = obs[i] 
	three = np.array([i]) 
	f = np.append(one, two) 
	f = np.append(f, three) 
	return f 


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


	score_logger = ScoreLogger(ENV_NAME)
	observation_space = SA_OBS_SPACE
	action_space = SA_ACTION_SPACE 
	dqn_solver = DQNSolver(observation_space, action_space)
	# run = 0 
	for i in xrange(episodes):
		print "==== Starting episode no:",i,"====","\n"
		# observation = env.reset()
		observation = ca_reset() 
		reward_per_episode = 0


		# run += 1
		obs = env.reset() 
		# step = 0 

		for t in xrange(steps):
			#rendering environment (optional)         
			env.render() 
			print "Step: ", t 

			x_arr = [] 
			observation_arr = [] 
			action_arr = [] 

			action_n = [] 

			state_arr = [] 
			next_state_arr = [] 
			action_n_arr = [] 

			# step += 1

			for z in range(env.n): 

				x = observation
				# x_arr.append(x) 
				x_arr.append(np.array(list(x))) 
				action = agent.evaluate_actor(np.reshape(x, [1, num_states])) 
				noise = exploration_noise.noise()
				action = action[0] + noise #Select action according to current policy and exploration noise 
				action_arr.append(action) 
				# print "Action at step", t ," :",action,"\n" 

				state = sa_state(action, obs, z)  
				state = np.reshape(state, [1, observation_space]) 
				state_arr.append(state) 

				act = dqn_solver.act(state) 
				a = np.zeros(SA_ACTION_SPACE) 
				a[act] = 1.0 
				action_n.append(a) 
				action_n_arr.append(act) 

				# print "CA State: ", x 
				# print "CA Action: ", action 
				# print "SA State: ", state 
				# print "SA Action: ", act 
				# print "------------------------------------------" 

				observation[z] = act 
				# observation_arr.append(observation) 
				observation_arr.append(np.array(list(observation)))

			next_obs, reward_n, done_n, info_n = env.step(action_n) 

			# for z in range(env.n): 
			# 	if reward_n[z] > -DONE_VALUE: 
			# 		done_n[z] = True 

			reward = reward_n[0] 
			done = all(done_n) 

			print 
			print "Reward_n: ", reward_n 
			print 

			for z in range(env.n): 
				ns = sa_state(action_arr[z], next_obs, z) 
				ns = np.reshape(ns, [1, observation_space]) 
				next_state_arr.append(ns) 



			# observation,reward,done,info=env.step(action)
			# observation,reward,done,info=ca_step(action) 

			for z in range(env.n): 
			    #add s_t,s_t+1,action,reward to experience memory
			    # print x_arr[z], observation_arr[z], action_arr[z], reward_n[z], done_n[z] 
			    agent.add_experience(x_arr[z], observation_arr[z], action_arr[z], reward_n[z], done_n[z]) 
			    dqn_solver.remember(state_arr[z], action_n_arr[z], reward_n[z], next_state_arr[z], done_n[z]) 

			obs = next_obs 


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
				np.savetxt('rewards/episode_reward.txt',reward_st, newline="\n")
				 
				print "Run: " + str(i) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(reward_per_episode/t) 
				score_logger.add_score(reward_per_episode/t, i) 
				print '\n\n'

				break
			dqn_solver.experience_replay() 
	
	total_reward+=reward_per_episode 
	print "Average reward per episode {}".format(total_reward / episodes)    
	print "+++++++++++++++++++++++++++++++++++++++++++++++++" 
	print 
	print 

      
    

if __name__ == '__main__':
    main()    


print "Hi" 