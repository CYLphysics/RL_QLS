from dwave_qbsolv import QBSolv
import numpy as np
import itertools
from time import time 
from tqdm.notebook import trange, tqdm
import matplotlib.pyplot as plt
import networkx as nx
from gym_example.envs.lssa_env import LSSA_v0

from ray.tune.registry import register_env
import gym
import ray
import ray.rllib.agents.ppo as ppo

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import ray.rllib.algorithms.impala as impala 

from util import *

stt = time()


approximation_ratio_list_rl_impala_list = []
approximation_ratio_list_rl_impala2_list = []
approximation_ratio_list_random_list = []

win_count_impala = 0
win_count_impala2 = 0
test_num = 3

ham_data = []
for test in range(test_num):
    # ======= generate random test Hamiltonian ======
    n_qubit = 128
    rand_weight = (np.random.rand( int(n_qubit*(n_qubit-1)/2) + n_qubit ) - 0.5)*2
    ham = reconstruct_ham_from_weight(rand_weight, n_qubit)
    ham_data.append(ham)

    # ===============================================


for test in range(test_num):

    ham = ham_data[test]
    # ====== random local update ======
    group_size_lo = 5
    res_local_optimization = LSSA_local_optimization(ham, group_size=group_size_lo, init_config="random", approximation_ratio=True, iteration=500, print_energy=False)
    approximation_ratio_list_random_list.append(res_local_optimization[1])
    now = time()
    print("Generating result of random LS, data ", test+1, " /",test_num, ", duration = ", now - stt, " s")
    np.array(approximation_ratio_list_random_list).dump("128_pick_5/approximation_ratio_list_random_list_test6.dat")

    # =============== Impala =================
    #ham = ham_data[test]
    ray.init(
        ignore_reinit_error = True,
        log_to_driver = False
        )
    agent_model = "impala"
    train = False
    select_env = "lssa-v0"
    #solver = "dwave-tabu", # "qaoa", "dwave-tabu", "exact"
    register_env(select_env, lambda config: LSSA_v0([ham], train))
    if agent_model == "ppo":
        config = ppo.DEFAULT_CONFIG.copy()
    elif agent_model == "impala":
        config = impala.DEFAULT_CONFIG.copy()
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
    n_qubit = count_n_qubit(ham_data[0])
    n_terms = int(n_qubit*(n_qubit - 1)*0.5 + n_qubit)
    spaces = {
        'graph': gym.spaces.Box(low=-9999, high=9999, shape=(n_qubit, n_qubit), dtype=np.float32),
        'configuration': gym.spaces.Box(low=0, high=1, shape=(n_qubit*history, ), dtype=np.int)
        }
    config["observation_space"] = gym.spaces.Dict(spaces)
    config["action_space"] = gym.spaces.MultiDiscrete([n_qubit for i in range(Ng)])
    config["use_critic"] = True 
    config['model']['fcnet_hiddens'] = [512, 256, 128, 64, 32]
    config['model']['fcnet_activation'] = "relu"
    config['model']['conv_filters'] = [[32, [4, 4], 4],[32, [4, 4], 2],]
    config["num_workers"] = 1
    if agent_model == "ppo":
        ppo_config = ppo.PPOConfig.from_dict(config)
        agent = ppo_config.build(env=select_env)
    elif agent_model == "impala":
        impala_config = impala.ImpalaConfig.from_dict(config)
        agent = impala_config.build(env=select_env)
    policy = agent.get_policy()
    model = policy.model
    #agent.restore("128_pick_5/20230218_1000ham_impala_2000iter/checkpoint_001903")
    #agent.restore("128_pick_5/20230218_1000ham_impala_2000iter_2nd_round/checkpoint_003372")
    agent.restore("128_pick_5/1000ham_impala_2000iter_3rd_round/checkpoint_003425")
    
   
    env = gym.make(select_env, hamiltonian_dataset = [ham], train = train)
    state = env.reset()
    sum_reward = 0
    n_step = 500
    reward_list = [] 
    approximation_ratio_impala_list = [] 
    for step in range(n_step):
        action = agent.compute_single_action(state)
        state, reward, done, info = env.step(action)
        approximation_ratio_impala_list.append(info['Approximation ratio'])
        sum_reward += reward
        reward_list.append(reward)
    approximation_ratio_list_rl_impala_list.append(approximation_ratio_impala_list)
    ray.shutdown()
    np.array(approximation_ratio_list_rl_impala_list).dump("128_pick_5/approximation_ratio_list_rl_impala_list_test6.dat")

    # ====== random vs impala ======
    now = time()
    if max(approximation_ratio_impala_list) >= max(approximation_ratio_list_random_list[test]):
        win_count_impala += 1
        print("win !, impala : ",max(approximation_ratio_impala_list)," random : ", max(approximation_ratio_list_random_list[test]) ,", current impala win count = ", win_count_impala, "/" , test +1, ", win rate = ", win_count_impala/( test +1), ", duration = ", now - stt, " s")
    elif max(approximation_ratio_impala_list) < max(approximation_ratio_list_random_list[test]):
        print("loss !, impala : ",max(approximation_ratio_impala_list)," random : ", max(approximation_ratio_list_random_list[test]) ,", current impala win count = ", win_count_impala, "/" , test +1, ", win rate = ", win_count_impala/( test +1), ", duration = ", now - stt, " s")
