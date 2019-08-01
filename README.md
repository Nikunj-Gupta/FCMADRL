# FULLY COOPERATIVE MULTI-AGENT DEEP REINFORCEMENT LEARNING 
This repository comprises of the code for my **Master's Thesis** work on leveraging the advantages of a **Multi-Agent Systems'** paradigm of **Centralised Learning and Decentralised Execution** in **Reinforcement Learnin**g to train a group of intelligent agents to learn to accomplish a task **cooperatively**. 

## Abstract 
Coordination of autonomous vehicles, automating warehouse management system or another real world complex problem like large-scale fleet management can be easily fashioned as cooperative multi-agent systems. Presently, algorithms in Reinforcement Learning (RL), which are designed for single agents, work poorly in multi-agent settings and hence there is a need for RL frameworks in Multi-Agent Systems (MAS). But, Multi-Agent Reinforcement Learning (MARL) poses its own challenges, some of the major ones being the problem of non-stationarity and the exponential growth of the joint action space with increasing number of agents. A possible solution to these complexities is to use Centralised learning and Decentralised execution of policies, however the question of using the notion of centralised learning to the fullest still remains open. As apart of this thesis, we developed an architecture, adopting the framework of centralised learning with decentralised execution, where all the actions of the individual agents are given as input to a central agent and it outputs information for them to utilize. Thus, the system of individual agents are given the opportunity of using some extra information (about other agents affecting the environment directly) from a central agent which also helps in easing their training. Results for the same are showcased on the Multi-Agent Particle Environment (MPE) by OpenAI. An extension of the architecture for the case of warehouse logistics is also shown in the thesis. 


## Environment 
The following environment was used- Multi-Agent Particle Environment 
Link: [**MPE**](https://github.com/openai/multiagent-particle-envs) 

## Repository Tour 
**Merge 1:** This directory has the code relevant to the architecture where we propose to adapt from Deep Determinitic Policy Gradients (DDPG) and Proximal Policy Gradients (PPO). 

**Merge 2:** This directory has the code relevant to the architecture where we propose to adapt from Deep Determinitic Policy Gradients (DDPG) and Deep Q-Networks (DQN). 


**Papers:** This folder contains an important subset of the papers explored during the thesis. 

## Adaptations 

[**DDPG**](https://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html) ; 

[**PPO**](https://github.com/uidilr/ppo_tf) 
