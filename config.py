ENV_NAME = "FCMARL" 

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.99 

FILE_PATH_DQN = "dqn/model/"
FILE_NAME_DQN = "model0.h5"
CHECKPOINT = 1
LOG_PATH = "logs/"

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
    # print ("Hello Simple_Spread Env") 
    return env 

env = make_env("simple_spread") 


#specify parameters here:
episodes=5 
steps=5 
CA_OBS_SPACE = env.n 
CA_ACTION_SPACE = 5 # env.observation_space[0].shape[0] 
SA_OBS_SPACE = CA_ACTION_SPACE + env.observation_space[0].shape[0] + 1 
SA_ACTION_SPACE = env.action_space[0].n 
CA_ACTION_BOUND = 5.0 
is_batch_norm = False #batch normalization switch
DONE_VALUE = 0.0 