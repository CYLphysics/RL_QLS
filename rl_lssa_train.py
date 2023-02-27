from dwave_qbsolv import QBSolv
import numpy as np
import csv
import itertools
from time import time 
import datetime
from tqdm.notebook import trange, tqdm
import matplotlib.pyplot as plt
import networkx as nx
import random 
import math
from scipy.optimize import minimize, basinhopping

from gym_example.envs.lssa_env import LSSA_v0

from ray.tune.registry import register_env
import gym
import os
import ray
import ray.rllib.agents.ppo as ppo
import shutil

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import ray.rllib.algorithms.impala as impala 


# ================ Generate training dataset ================

n_qubit = 128

def reconstruct_ham_from_weight(weight, N):

    reg = list(range(N))
    terms = []; weights = []
    for qubit in reg:
        terms.append([qubit, ])
        weights.append(0)
    for q1, q2 in itertools.combinations(reg, 2):
        terms.append([q1, q2])
        weights.append(0)

    empty_ham = {tuple(term):weight for term,weight in zip(terms,weights)}
    
    index = 0 
    for term in empty_ham.items():
        empty_ham[term[0]] = weight[index]
        index += 1

    return empty_ham  

ham_dataset = []

for i_ in range(1000):    
    rand_weight = (np.random.rand( int(n_qubit*(n_qubit-1)/2) + n_qubit ) - 0.5)*2
    ham = reconstruct_ham_from_weight(rand_weight, n_qubit)
    ham_dataset.append(ham)

# ================================================================

# ================ training the RL agent ================

train_reward_list_min_list = [] 
train_reward_list_mean_list = []
train_reward_list_max_list = []
train_reward_list = []

for train_num in range(1):
    print("============= generation", train_num+1, "=============")
    # init directory in which to save checkpoints
    chkpt_root = "128_pick_5/1000ham_impala_2000iter_3rd_round"

    shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)

    # init directory in which to log results
    ray_results = "{}/ray_results/".format(os.getenv("HOME"))
    shutil.rmtree(ray_results, ignore_errors=True, onerror=None)


    # start Ray -- add `local_mode=True` here for debugging
    ray.init(
        ignore_reinit_error = True,
        #num_gpus=2
        #log_to_driver = False
        )

    # register the custom environment
    select_env = "lssa-v0"
    #select_env = "fail-v1"   
    train = True # specify this is the training process or testing process
    register_env(select_env, lambda config: LSSA_v0(ham_dataset, train)) 
    # How to import multi-Hamiltonian information into the environment ? 
    # ham = a set of Hamiltonians ? 

    #register_env(select_env, lambda config: Fail_v1())

    agent_model = "impala" # "ppo", "impala", "dqn", "apex_dqn"

    # configure the environment and create agent

    if agent_model == "ppo":
        config = ppo.DEFAULT_CONFIG.copy()
    elif agent_model == "impala":
        config = impala.DEFAULT_CONFIG.copy() # although the reward looks nice, the testing result is bad 

    #config["env_config"] = LSSA_v0().hamiltonian_assign(ham)
    config["log_level"] = "ERROR"  #"WARN"
    config["framework"] = "torch"

    history = 1
    Ng = 5

    def count_n_qubit(ham):
        bias_list = []
        coupling_max_index_list = [] 

        for term in ham:
            if len(term) == 1:
                bias_list.append(term[0])
            elif len(term) != 1:
                coupling_max_index_list.append(max(term))
                #break;
        n_qubit = max([max(coupling_max_index_list),max(bias_list)]) +1 
        return n_qubit

    def term_to_list(ham, n_qubit):
        term_list = [0 for i in range(int(n_qubit*(n_qubit - 1)*0.5 + n_qubit))]
        for i, j in enumerate(ham):
            term_list[i] += ham[j]
        return term_list
    
    n_qubit = count_n_qubit(ham_dataset[0])
    n_terms = int(n_qubit*(n_qubit - 1)*0.5 + n_qubit)

    spaces = {
        'graph': gym.spaces.Box(low=-9999, high=9999, shape=(n_qubit, n_qubit), dtype=np.float32),
        'configuration': gym.spaces.Box(low=0, high=1, shape=(n_qubit*history, ), dtype=np.int)
        }

    config["observation_space"] = gym.spaces.Dict(spaces)


    config["action_space"] = gym.spaces.MultiDiscrete([n_qubit for i in range(Ng)])
    #config["sgd_minibatch_size"] = 128 #32 #128
    #config["num_sgd_iter"] = 32 #32 
    config["use_critic"] = True 

    config['model']['fcnet_hiddens'] = [512, 256, 128, 64, 32]

    #[1024, 512, 256, 128, 64, 32] # -> for Np = 128
    #[512, 256, 128, 64, 32] # -> for Np = 32, 64
    
    config['model']['fcnet_activation'] = "relu"
    config['model']['conv_filters'] = [[32, [4, 4], 4], [32, [4, 4], 2],]
    #[[32, [4, 4], 4], [32, [4, 4], 2],] #for Np = 32, 64 
    #[[64, [8, 8], 8], [32, [4, 4], 4],]  # for Np = 128

    #config["num_envs_per_worker"] = 4
    #config['lr'] = 1e-7 #1e-8
    #config['lr_schedule'] = [
    #    [0, 1e-7],
    #    [10000, 5e-8],
    #    [20000, 1e-8]
    #]
    config["num_workers"] = 68
    #config["num_gpus"] = 2
    #config["train_batch_size"] = 16 
    #config["num_sgd_iter"] = 10
    #config["sgd_minibatch_size"] = 16

    # 'model': {'_use_default_native_models': False, '_disable_preprocessor_api': False, '_disable_action_flattening': False, 'fcnet_hiddens': [256, 256], 'fcnet_activation': 'tanh', 'conv_filters': None, 'conv_activation': 'relu', 'post_fcnet_hiddens': [], 'post_fcnet_activation': 'relu', 'free_log_std': False, 'no_final_linear': False, 'vf_share_layers': False, 'use_lstm': False, 'max_seq_len': 20, 'lstm_cell_size': 256, 'lstm_use_prev_action': False, 'lstm_use_prev_reward': False, '_time_major': False, 'use_attention': False, 'attention_num_transformer_units': 1, 'attention_dim': 64, 'attention_num_heads': 1, 'attention_head_dim': 32, 'attention_memory_inference': 50, 'attention_memory_training': 50, 'attention_position_wise_mlp_dim': 32, 'attention_init_gru_gate_bias': 2.0, 'attention_use_n_prev_actions': 0, 'attention_use_n_prev_rewards': 0, 'framestack': True, 'dim': 84, 'grayscale': False, 'zero_mean': True, 'custom_model': None, 'custom_model_config': {}, 'custom_action_dist': None, 'custom_preprocessor': None, 'lstm_use_prev_action_reward': -1}
    # 'sgd_minibatch_size': 128, 'num_sgd_iter': 30,

    if agent_model == "ppo":
        #agent = ppo.PPOTrainer(config, env=select_env)  
        ppo_config = ppo.PPOConfig.from_dict(config)
        agent = ppo_config.build(env=select_env)

    elif agent_model == "impala":
        impala_config = impala.ImpalaConfig.from_dict(config)
        agent = impala_config.build(env=select_env)


    status = "Iteration {:2d}, reward min : {:6.2f}/ mean : {:6.2f}/ max : {:6.2f} len {:4.2f} save ? {}"
    n_iter = 2000

    train_reward_list_min = []
    train_reward_list_mean = []
    train_reward_list_max = []  
    train_reward = [] 
    best_mean_reward_so_far = []

    # retrieve a trained agent to resume the training ~! 
    #agent.restore("128_pick_5/20230218_1000ham_impala_2000iter/checkpoint_001903")
    agent.restore("128_pick_5/1000ham_impala_2000iter_2nd_round/checkpoint_003372")
    # train a policy with RLlib using PPO
    for n in range(n_iter):
        result = agent.train()
        
        # should only save the agent with good performance
        
        train_reward_list_min.append(result["episode_reward_min"])
        train_reward_list_mean.append(result["episode_reward_mean"])
        train_reward_list_max.append(result["episode_reward_max"])

        train_reward += result["hist_stats"]['episode_reward']

        if result["episode_reward_mean"] in np.sort(train_reward_list_mean)[::-1][:10]:
            chkpt_file = agent.save(chkpt_root) 

            print(status.format(
                    n + 1,
                    result["episode_reward_min"],
                    result["episode_reward_mean"],
                    result["episode_reward_max"],
                    result["episode_len_mean"],
                    chkpt_file
                    ))
                    
        elif result["episode_reward_mean"] not in np.sort(train_reward_list_mean)[::-1][:5]:

            print(status.format(
                    n + 1,
                    result["episode_reward_min"],
                    result["episode_reward_mean"],
                    result["episode_reward_max"],
                    result["episode_len_mean"],
                    " not saving as it is not the best 5 so far"
                    ))

        train_reward_list_min_list.append(train_reward_list_min)
        train_reward_list_mean_list.append(train_reward_list_mean)
        train_reward_list_max_list.append(train_reward_list_mean)
        train_reward_list.append(train_reward_list_mean)
    
        np.array(train_reward_list_min_list).dump(chkpt_root+"_train_reward_list_min_list.dat")
        np.array(train_reward_list_mean_list).dump(chkpt_root+"_train_reward_list_mean_list.dat")
        np.array(train_reward_list_max_list).dump(chkpt_root+"_train_reward_list_max_list.dat")
        np.array(train_reward).dump(chkpt_root+"_train_reward.dat")

    ray.shutdown()

# in the case 64_pick_5, cn8 needs about XX mins to start training
# in the case 128_pick_5, cn8 needs about 30 mins to start training, takes ~938 min to train 2000 iterations 
# ================================================================
