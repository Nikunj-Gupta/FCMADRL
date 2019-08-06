import gym
from gym.spaces import Box, Discrete
import numpy as np
import random
import logging

from ddpg.ddpg import DDPG
from ddpg.ou_noise import OUNoise

from dqn.dqn import DQNSolver
from dqn.scores.score_logger import ScoreLogger

from config import *


def ca_reset():
    return np.random.choice(SA_ACTION_SPACE, CA_OBS_SPACE)



class FCMADRL:
    def __init__(self):
        self.observation_space = SA_OBS_SPACE
        self.action_space = SA_ACTION_SPACE
        # self.agent = agent
        self.agent = DDPG(env, is_batch_norm, CA_OBS_SPACE, CA_ACTION_SPACE, CA_ACTION_BOUND)
        self.dqn_solver = DQNSolver(SA_OBS_SPACE, SA_ACTION_SPACE)

        logging.basicConfig(file_name="logs/log.log", format='%(asctime)s %(message)s', file_mode='w+')
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

    def use_existing_dqn(self, dqn_model):
        self.dqn_solver.model = dqn_model

    def get_ddpg(self):
        return self.agent

    def get_dqn(self):
        return self.dqn_solver

    def get_dqn_model(self, dqn_solver):
        return dqn_solver.model

    """ 
    ca_step() is just for testing purposes 
    """

    def ca_step(self, action):
        return np.random.choice(SA_ACTION_SPACE, CA_OBS_SPACE), np.random.choice(10), np.random.choice(
            [True, False]), {}

    """ 
    sa_state(): To merge the two states received by the individual agents 
    (one from central agent and one from the environment) into one vector 
    """

    def sa_state(self, x, obs, i):
        one = x
        two = obs[i]
        three = np.array([i])
        f = np.append(one, two)
        f = np.append(f, three)
        return f

    def fcmadrl(self):
        # Randomly initialize critic,actor,target critic, target actor network and replay buffer
        exploration_noise = OUNoise(CA_ACTION_SPACE)
        counter = 0
        reward_per_episode = 0
        total_reward = 0
        num_states = CA_OBS_SPACE
        num_actions = CA_ACTION_SPACE

        self.logger.debug("Number of States:"+str(num_states))
        self.logger.debug("Number of Actions:"+str(num_actions))
        self.logger.debug("Number of Steps per episode:"+str(steps))
        # saving reward:
        reward_st = np.array([0])

        score_logger = ScoreLogger(ENV_NAME)

        # run = 0
        for i in xrange(episodes):
            print "==== Starting episode no:", i, "====", "\n"
            # observation = env.reset()
            observation = ca_reset()
            reward_per_episode = 0

            # run += 1
            obs = env.reset()
            # step = 0

            for t in xrange(steps):
                # rendering environment (optional)
                env.render()
                print "Step: ", t

                x_arr = []
                observation_arr = []
                action_arr = []
                action_n = []
                state_arr = []
                next_state_arr = []
                action_n_arr = []

                for z in range(env.n):
                    self.take_action(action_arr, action_n, action_n_arr, exploration_noise, num_states, obs,
                                     observation, observation_arr, state_arr, x_arr, z)

                next_obs, reward_n, done_n, info_n = env.step(action_n)

                reward = reward_n[0]
                done = all(done_n)
                print "Reward_n: ", reward_n

                self.update_next_state(action_arr, next_obs, next_state_arr)
                self.memory_store(action_arr, action_n_arr, done_n, next_state_arr, observation_arr, reward_n,
                                  state_arr, x_arr)
                obs = next_obs
                # train critic and actor network
                if counter > 64:
                    self.agent.train()
                reward_per_episode += reward
                counter += 1
                # check if episode ends:
                if done or (t == steps - 1):
                    print 'EPISODE: ', i, ' Steps: ', t, ' Total Reward: ', reward_per_episode
                    print "Printing reward to file"
                    exploration_noise.reset()  # reinitializing random noise for action exploration
                    reward_st = np.append(reward_st, reward_per_episode)
                    np.savetxt('rewards/episode_reward.txt', reward_st, newline="\n")

                    print "Run: " + str(i) + ", exploration: " + str(
                        self.dqn_solver.exploration_rate) + ", score: " + str(reward_per_episode / t)
                    score_logger.add_score(reward_per_episode / t, i)
                    print '\n\n'

                    break
                self.dqn_solver.experience_replay()
            if (i % CHECKPOINT == 0):
                self.dqn_solver.save_dqn_model(i)

        total_reward += reward_per_episode
        print "Average reward per episode {}".format(total_reward / episodes)
        return total_reward

    def update_next_state(self, action_arr, next_obs, next_state_arr):
        for z in range(env.n):
            ns = self.sa_state(action_arr[z], next_obs, z)
            ns = np.reshape(ns, [1, self.observation_space])
            next_state_arr.append(ns)

    def take_action(self, action_arr, action_n, action_n_arr, exploration_noise, num_states, obs, observation,
                    observation_arr, state_arr, x_arr, z):
        action = self.get_message(action_arr, exploration_noise, num_states, observation, x_arr)
        state = self.sa_state(action, obs, z)
        state = np.reshape(state, [1, self.observation_space])
        state_arr.append(state)
        act = self.get_final_action(action_n, action_n_arr, state)
        self.logger.debug("SA_Action: "+str(act))
        # print "CA State: ", x
        # print "CA Action: ", action
        # print "SA State: ", state
        # print "SA Action: ", act
        observation[z] = act
        observation_arr.append(np.array(list(observation)))

    def memory_store(self, action_arr, action_n_arr, done_n, next_state_arr, observation_arr, reward_n, state_arr,
                     x_arr):
        for z in range(env.n):
            # add s_t,s_t+1,action,reward to experience memory
            # print x_arr[z], observation_arr[z], action_arr[z], reward_n[z], done_n[z]
            self.agent.add_experience(x_arr[z], observation_arr[z], action_arr[z], reward_n[z], done_n[z])
            self.dqn_solver.remember(state_arr[z], action_n_arr[z], reward_n[z], next_state_arr[z], done_n[z])

    def get_final_action(self, action_n, action_n_arr, state):
        act = self.dqn_solver.act(state)
        a = np.zeros(SA_ACTION_SPACE)
        a[act] = 1.0
        action_n.append(a)
        action_n_arr.append(act)
        return act

    def get_message(self, action_arr, exploration_noise, num_states, observation, x_arr):
        x = observation
        # x_arr.append(x)
        x_arr.append(np.array(list(x)))
        action = self.agent.evaluate_actor(np.reshape(x, [1, num_states]))
        noise = exploration_noise.noise()
        action = action[0] + noise  # Select action according to current policy and exploration noise
        action_arr.append(action)
        self.logger.debug("Action at Step: " + str(action))
        # print "Action at step", t ," :",action,"\n"
        return action


if __name__ == '__main__':
    # main()
    mas = FCMADRL()
    total_reward = mas.fcmadrl()
    print "Final: ", total_reward
    print "Hi"
