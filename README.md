# FULLY COOPERATIVE MULTI-AGENT DEEP REINFORCEMENT LEARNING 
This repository comprises of the code for my **Master's Thesis** work on leveraging the advantages of a **Multi-Agent Systems'** paradigm of **Centralised Learning and Decentralised Execution** in **Reinforcement Learning** to train a group of intelligent agents to learn to accomplish a task **cooperatively**. 

## Abstract 
Coordination of autonomous vehicles, automating warehouse management system or another real world complex problem like large-scale fleet management can be easily fashioned as cooperative multi-agent systems. Presently, algorithms in Reinforcement Learning (RL), which are designed for single agents, work poorly in multi-agent settings and hence there is a need for RL frameworks in Multi-Agent Systems (MAS). But, Multi-Agent Reinforcement Learning (MARL) poses its own challenges, some of the major ones being the problem of non-stationarity and the exponential growth of the joint action space with increasing number of agents. A possible solution to these complexities is to use Centralised learning and Decentralised execution of policies, however the question of using the notion of centralised learning to the fullest still remains open. As apart of this thesis, we developed an architecture, adopting the framework of centralised learning with decentralised execution, where all the actions of the individual agents are given as input to a central agent and it outputs information for them to utilize. Thus, the system of individual agents are given the opportunity of using some extra information (about other agents affecting the environment directly) from a central agent which also helps in easing their training. Results for the same are showcased on the Multi-Agent Particle Environment (MPE) by OpenAI. An extension of the architecture for the case of warehouse logistics is also shown in the thesis. 


## Environment 
The following environment was used- Multi-Agent Particle Environment 
This environment needs to be downloaded and installed in order reproduce the results. 
Follow the instructions in the following [**here**](https://github.com/openai/multiagent-particle-envs) 
FCMADRL uses only the Cooperative Navigation (simple_spread.py) env from the set of environments in MPE. 
The parameters can be very problem specific. In order to change the number of agents or landmarks, you must go to *< MPE root directory >/multiagent/scenarios/simple_spread.py* manually. 


## USAGE 

The package can be downloaded using pip: `pip install FCMADRL==<version>` 
Small example: 
```python
from FCMADRL import framework, inference
``` 

## Code structure 
[*config.py*](https://github.com/Nikunj-Gupta/FCMADRL/blob/master/config.py): This is the configuration file. It contains all the parameters required by the various networks in the repository. This becomes quite useful when hyperparameters for various runs need to modified. 

[*ddpg/ddpg.py*](https://github.com/Nikunj-Gupta/FCMADRL/blob/master/ddpg/ddpg.py): ddpg stands for Deep Deterministic Policy Gradients. It has the network for the central agent (because it has to deal with a output in the continuous space). 

[dqn/dqn.py](https://github.com/Nikunj-Gupta/FCMADRL/blob/master/dqn/dqn.py): dqn stands for Deep Q-Network. This file has the network for the individual agents. 

[*framework.py*](https://github.com/Nikunj-Gupta/FCMADRL/blob/master/framework.py): This is the main function. It incorporates both the components and connects them to the environment. 

**Papers:** This folder contains an important subset of the papers explored during the thesis. 

## Running the code 
If you have docker, you can use the following dockerfile to build the environment to use my code as is. 
* [**Dockerfile**](https://github.com/Nikunj-Gupta/FCMADRL/blob/master/dockerfile) 
Keep the reuirements.txt along with the dockerfile. 

* Clone the repository. 
* Clone the repository of the environment and install it as described in the instructions of the same. 
* run framework.py using the following command 
`python framework.py` 

The code shall start running and it will stop when it has achieved its objective (all agents covering all the landmarks) for 100 episodes straight. 

## Adaptations 
[**DDPG**](https://github.com/stevenpjg/ddpg-aigym)
[**DQN**](https://github.com/gsurma/cartpole) 
