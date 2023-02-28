# Reinforcement Learning Quantum Local Search 

<img src="/images/architec.png" width="800px" align="center">

## Project description 
[Quantum Local Search](https://doi.org/10.1002/qute.201900029) utilizes small quantum computers to solve large combinatorial optimization problems by performing local search on quantum hardware, given an initial start point. However, the random selection process of the sub-problem to solve may not be efficient. In this hackathon project, we aim to train a reinforcement learning (RL) agent to find a better strategy for choosing the subproblem to solve in the quantum local search scenario, rather than relying on random selection.

In this repository, we provide the source code for implementing Reinforcement Learning Quantum Local Search (RL-QLS). First, to train an RL agent, an environment for the agent is required. The folder `gym_example/envs/` contains a file named `lssa_env.py`, which describes the gym environment of a QLS task. The file includes the basic elements for an environment, such as `__init__()`, `reset()` and `step()`. These elements enable the agent to explore and search for better strategies in the game (i.e., the choice of sub-problem). Here, "better" refers to a higher reward value obtained by the agent. The reward value is set as the approximation ratio of the solution configuration.

Secondly, once the environment is set up, the agent can be trained using the code provided in `rl_lssa_train.py`, which uses the IMPALA method. The code allows for the saving of the agent checkpoint at a specific training iteration. The file containing the trained agent checkpoint is not fully provided due to GitHub's file size limitations. However, it is possible to reproduce the results using the code provided in this repository.

Finally, after training an agent, the next step is to load the model and apply a testing scenario. In `rl_lssa_test.py`, for each trained agent, we generate 100 fully-connected random Ising models as testing data to find the low-energy solution and compare the results with the QLS random sub-problem picking strategy. The only provided trained agent model here is the 32_pick_5 scenario agent, since it is the only scenario that the saved model is less than 25MB. 

In the `RL_LSSA_plot.ipynb` notebook, the mean episode reward during the training process and the approximation ratio during the testing steps are plotted for three scenarios: a 32-variable problem with a 5-variable solver, a 64-variable problem with a 5-variable solver, and a 128-variable problem with a 5-variable solver. In all of these cases, the RLQLS method provides superior results. It not only reaches a better approximation ratio configuration faster, but also achieves a better approximation ratio at the end of testing.

<img src="/images/reward.png" width="800px" align="center">
<img src="/images/Approx.png" width="800px" align="center">

## Future work 

* ### Increase the subsystem size to be solved  
    For now, the subsystem size is 5, which could be scaled up in the future, with possible improvement of RL-QLS process, since the RL agent handles larger sub-problem at a time. 
* ### Applicability of specific problem  
    Can we train on fully-connected random graph and apply the RL agent to the regular-￼ graph combinatorial optimization problems ? or addition training is required ?
* ### Transferability of different models  
    For now, each problem size and sub-problem size requires different RL model, can these models share some common “knowledge” and reduce the required training time ? 
