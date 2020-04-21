# README
This repo contains my solution for the second project of the udacity Deep Reinforcement Learning Nanodegree.
The code and implementation is based on the DDPG example code given in the udacity DRL Nano Degree (for the Open AI pendulum task) and modified for the Unity ML Continuous Control Task.


# Short Description of the environment

The environment is the Unity ML Reacher environment [(link)](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher). It describes a double jointed arm. The goal of the agent is to move the arm to and stay in a given target location. The agent receives a reward of +0.1 for every timestep it is in the target location. 

There are two variants provided for this task, one contains a single arm, the other 20 identical copies for distributed learning. The provided solution in this repo solves the version with a single arm. 

## Observation Space

The observation space consists of 33 continuous variables corresponding to position, rotation, velocity, and angular velocities of the two arms.

## Action Space

The action space is continuous and 4-dimensional, corresponding to the torque applicable to the two joints (for the two degrees of freedom on each joint).

## Solution criterion

The environment is considered to be solved, if the agent scores on average +30 on 100 consecutive episodes. 


# Files in the Repository

The files of interest in the repo are: 

- `Continuous_Control.ipynb`: Notebook used to control and train the agent. The entry point to the code. 
- `DDPG_agent.py`: Create an Agent class that interacts with and learns from the environment 
- `models.py`: Contains the two networks for "actor" and "critic". values 
- `checkpoint_actor.pth`: Saved weights for actor network
- `checkpoint_critic.pth`: Saved weights for the critic network
- `report.pdf`: Project report including a short introduction of the DDGP algorithm used, the hyperparameters and a short discussion of the results. 


# Getting Started, Installation and Dependencies

To run this code, you have to install the dependencies and the environment.

## Dependencies  

The code requires Python 3. The  necessary dependencies can be found in `./python/requirements.txt` 
Batch installation is done like so: 
```
pip install ./python/requirements.txt
``` 
## Environment

The necessasry Unity environment can be downloaded from the following locations:

- Linux: [(link)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
- Mac OSX: [(link)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
- Windows (64-bit): [(link)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)



## Execution

Once you have cloned this repository and istalled the dependencies and the environment, the main entry point for execution is the Jupyter-notebook `Continuous_Control.ipynb`.



# Reference and Credits

The implementation is based on code from the the DDPG example given in the udacity DRL repo and only minimally adopted to the new environment. The complete list of references can be found in the project report.
