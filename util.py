from dwave_qbsolv import QBSolv
import numpy as np
import itertools


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


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

def LSSA_local_optimization(
    ham,
    group_size = 5,
    solver = "dwave-tabu",
    iteration = 200,
    print_energy = True,
    init_config = "random",
    approximation_ratio = False
):
    group_size = group_size

    bias_list = []
    coupling_max_index_list = [] 

    for i, term in enumerate(list(ham.keys())):
        if len(term) == 1:
            bias_list.append(term[0])
        elif len(term) != 1:
            coupling_max_index_list.append(max(term))
            #break;
    n_qubit = max([max(coupling_max_index_list),max(bias_list)]) +1 
    #print(n_qubit)

    if init_config == "random":
        init_config = np.random.choice([0,1], n_qubit)

    #print("init_config : ",init_config)

    if approximation_ratio == True:
        dwave_sol_res = Dwave_sol(ham)[1]

    energy_list_LSSA = [] 
    for trial in range(iteration):
        # random subgroup 
        subgroup_index = np.random.choice(list(range(n_qubit)), group_size, replace = False)
        ham_sub = {} 
        for term, weight in ham.items() : 
            if len(term) == 1:
                if term[0] in subgroup_index : 
                    ham_sub[term] = weight

            elif len(term) == 2:
                if term[0] in subgroup_index and term[1] in subgroup_index :
                    ham_sub[term] = weight

        if solver == "dwave-tabu":
            res_sub = Dwave_sol(ham_sub)
        elif solver == "exact":
            res_sub = exact_sol(ham_sub)
        """    
        elif solver == "qaoa":
            res_sub = QAOA_calculate(
                backend = backend,
                ham = ham_sub,
                optimizer_maxiter = optimizer_maxiter,
                shots = shots,
                print_eval = True
                )
        """
        if trial == 0:
            config_now = init_config

        config_temp = []
        for i in config_now:
            config_temp.append(i)

        for j, index in enumerate(subgroup_index):
            config_temp[index] = res_sub[0][0][j]

        energy_original = energy_finder(ham, config_now)
        energy_proposed = energy_finder(ham, config_temp)

        if energy_proposed < energy_original:
            config_now = config_temp
        elif energy_proposed > energy_original and np.random.rand() < np.exp(-100*(energy_proposed - energy_original)/abs(dwave_sol_res)):
            config_now = config_temp

        if print_energy == True:
            print("energy : ",energy_original)
            if approximation_ratio == True:
                dwave_sol_res = Dwave_sol(ham)[1]
                print("approximation ratio : ", energy_original/dwave_sol_res)
        energy_list_LSSA.append(energy_original)


    if approximation_ratio == False:
        return min(energy_list_LSSA), energy_list_LSSA, config_now
    elif approximation_ratio == True:
        dwave_sol_res = Dwave_sol(ham)[1]
        return min(energy_list_LSSA)/dwave_sol_res, np.array(energy_list_LSSA)/dwave_sol_res, config_now

def Dwave_sol(ham): # classical tabu solver

    n_qubit_ = 0
    for i_, term_ in enumerate(list(ham.keys())):
        if len(term_) == 1:
            n_qubit_+=1 
    
    # This is fine for fully connected graph
    index_couple_list = itertools.combinations(list(range(n_qubit_)), 2)
    h = dict(zip(list(range(n_qubit_)),list(ham.values())[:n_qubit_]))
    J = dict(zip(index_couple_list, list(ham.values())[n_qubit_:]))
    
    response = QBSolv().sample_ising(h, J)

    config = []
    for index, item in list(response.samples())[0].items():
        config.append(item)
        
    return [(np.array(config)*-1 +1)/2], list(response.data_vectors['energy'])[0]

def energy_finder(ham, test_config):

    config = ((np.array(test_config)*2)-1)*-1

    terms = list(ham.keys())
    weights = list(ham.values())

    energy = 0

    for i, term in enumerate(terms):
        if len(term) == 1:
            energy += config[term[0]]*weights[i]
        elif len(term) == 2:
            energy += config[term[0]]*config[term[1]]*weights[i]

    return energy 
