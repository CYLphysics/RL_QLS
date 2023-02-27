import gym
from gym.utils import seeding
import numpy as np
import random
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)

from dwave_qbsolv import QBSolv

from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.problems import QuadraticProgram
from qiskit.algorithms.optimizers import NELDER_MEAD
from qiskit.algorithms import QAOA
import itertools
import networkx as nx 
from qiskit.utils import QuantumInstance
from qiskit.providers.aer import QasmSimulator
from docplex.mp.model import Model
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer
from qiskit.providers.aer import *
from scipy.optimize import minimize

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

class LSSA_v0 (gym.Env):

    # land on the GOAL position within MAX_STEPS steps
    MAX_STEPS = 500 # 500 for Np = 128 # 200 for Np = 32, 64 

    metadata = {
        "render.modes": ["human"]
        }


    def __init__ (self, hamiltonian_dataset, train):

        self.history = 1
        self.train = train
        self.hamiltonian_dataset = hamiltonian_dataset
        
        self.Ng = 5
        
        
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

        self.n_qubit = count_n_qubit(self.hamiltonian_dataset[0])
        self.n_terms = int(self.n_qubit*(self.n_qubit - 1)*0.5 + self.n_qubit)
        self.action_space = gym.spaces.MultiDiscrete([self.n_qubit for i in range(self.Ng)])


        spaces = {
        'graph': gym.spaces.Box(low=-9999, high=9999, shape=(self.n_qubit,self.n_qubit), dtype=np.float32),
        'configuration': gym.spaces.Box(low=0, high=1, shape=(self.n_qubit*self.history, ), dtype=np.int)
        }
        self.observation_space  = gym.spaces.Dict(spaces)


        self.reset()


    def reset (self):
    
        
        #self.position = self.np_random.choice(self.init_positions)
        self.count = 0
        self.Approximation_ratio = 0  # reset the approximatino ratio


        picked_index = random.choice(list(range(len(self.hamiltonian_dataset))))
        #print("picked_index = ", picked_index)
        self.hamiltonian  = self.hamiltonian_dataset[picked_index]
        self.reference_sol = Dwave_sol(self.hamiltonian)[1]

        def term_to_list(ham, n_qubit):
            term_list = [0 for i in range(int(n_qubit*(n_qubit - 1)*0.5 + n_qubit))]
            for i, j in enumerate(ham):
                term_list[i] += ham[j]
            return term_list

        #self.hamiltonian_information = term_to_list(self.hamiltonian, self.n_qubit)

        
        self.hamiltonian_information = nx.to_numpy_array(
            graph_from_hamiltonian(self.hamiltonian)
            )
        # ===== prepare for initial state ===== 


        # for this environment, state is :
        # Hamiltonian information + historical configuration + historical costfunction + historical action

        self.state = {
            'graph': self.hamiltonian_information,
            'configuration': np.array([random.choice([0,1]) for i in range(self.n_qubit*self.history)])
        }

        self.reward = 0
        self.done = False
        self.info = {}

        return self.state


    def step (self, action):

        if self.train == True and self.done == True:
            # code should never reach this point
            print("EPISODE DONE!!!")

        elif self.train == True and self.count == self.MAX_STEPS:
                self.done = True;

        elif self.train == True and self.Approximation_ratio >= 0.95:
            print("Approximation ratio >= 0.95, Done !")
            self.done = True;

        
        else:
            assert self.action_space.contains(action)


            self.count += 1

            configuration_list   = self.state["configuration"][:self.n_qubit*self.history]
            
            # ==== update of the next step ========
            generated_group_index_list = action

            res = LSSA_local_update(
                ham = self.hamiltonian,
                train = self.train,
                group_index_list = generated_group_index_list,
                current_configuration = configuration_list[0:self.n_qubit],
                reference_sol = self.reference_sol, 
                solver = "dwave-tabu") # "vqe", "exact", "dwave-tabu"
            
            self.reward,  new_config = res

            configuration_list = np.insert(configuration_list, 0, list(np.array(new_config).flatten()))
            self.Approximation_ratio = energy_finder(self.hamiltonian, new_config) / self.reference_sol
            self.info["Approximation ratio"] = self.Approximation_ratio

            configuration_list = configuration_list[0:self.history*self.n_qubit]

            self.state = {
                'graph': self.hamiltonian_information,
                'configuration': np.array(configuration_list)
            }

        return [self.state, self.reward, self.done, self.info]


    def render (self, mode="human"):
        s = "position: {:2d}  reward: {:2d}  info: {}"
        print(s.format(self.state, self.reward, self.info))


    def seed (self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def close (self):
        pass



# ======= LSSA functions =======


ibmq_qasm_simulator = QasmSimulator(
    method='statevector', #matrix_product_state
    max_parallel_experiments = 0,

    )

cuQuantum_support = True

if cuQuantum_support == True:
    aer_simu_GPU = AerSimulator(method='statevector', device='GPU', cuStateVec_enable=True)


def LSSA_local_update(
        ham,
        train,
        group_index_list,
        current_configuration,
        reference_sol,
        solver = "dwave-tabu",
        backend = ibmq_qasm_simulator, # only take effect if solver == "vqe", # aer_simu_GPU
        shots    = 1024, # only take effect if solver == "vqe"
        optimizer_maxiter = 5 # only take effect if solver == "vqe"
        ):


    init_config = current_configuration

    #print("current config : ",init_config)

    #remove the repeated index in group_index_list
    group_index_list_not_repeat = [] 
    for i_ in group_index_list:
        if i_ not in group_index_list_not_repeat:
            group_index_list_not_repeat.append(i_)


    subgroup_index = group_index_list_not_repeat


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
    elif solver == "vqe":
        res_sub = VQE_calculate(
            backend = backend, # 
            ham = ham_sub,
            optimizer_maxiter = optimizer_maxiter,
            shots = shots,
            print_eval = True
            )
        
    config_now = init_config

    config_new = []
    for i in config_now:
        config_new.append(i)

    #print("subgroup_index = ", subgroup_index)
    for j, index in enumerate(subgroup_index):
        config_new[index] = res_sub[0][0][j]

    energy_original = energy_finder(ham, config_now)
    energy_proposed = energy_finder(ham, config_new)
    scale_factor = 100
    
    if train == True:
    
        #delta_E = scale_factor*(energy_original - energy_proposed)/abs(reference_sol)
        #if delta_E < 0:
        #    delta_E = delta_E  # penalty 

        #return delta_E, config_new 
        return energy_proposed/reference_sol, config_new

    if train == False:

        if energy_proposed < energy_original: # update ! 
            config_now = config_new
            energy_original = energy_proposed
            #print("update! ")

        # update with probability 
        elif energy_proposed > energy_original and np.random.rand() < np.exp(-100*(energy_proposed - energy_original)/abs(reference_sol)):
            #print("------------------------------------")
            #print("Delta E : ", (energy_proposed - energy_original)/abs(reference_sol))
            #print("Prob : ", np.exp(-100*(energy_proposed - energy_original)/abs(reference_sol)))
            #print("------------------------------------")

            config_now = config_new
            energy_original = energy_proposed
        
        return energy_original/(reference_sol), config_now 


def graph_from_hamiltonian(hamiltonian):

    G = nx.Graph()
    for term,weight in hamiltonian.items():
        if(len(term)==1):
            G.add_node(term[0], weight=weight)
            #G.add_edge(term[0], term[0], weight=weight) #if you want the qubo matrix
        elif(len(term)==2):
            G.add_edge(term[0], term[1], weight=weight)
    return G

    #return energy_original/(reference_sol), config_now 


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



def exact_sol(ham): #Exact diagonalization 

    terms = list(ham.keys())
    weights = list(ham.values())
    G = graph_from_hamiltonian(ham)
    register = list(np.sort(nx.nodes(G)))

    diag = np.zeros((2**len(register)))
    for i, term in enumerate(terms):
        out = np.real(weights[i])
        for qubit in register:
            if qubit in term:
                out = np.kron([1, -1], out)
            else:
                out = np.kron([1, 1], out)
                
        diag += out
    energy = np.min(diag)
    indices = []
    for idx in range(len(diag)):
        if diag[idx] == energy:
            indices.append(idx)
    config_strings = [np.binary_repr(index, len(register))[::-1] for index in indices]
    configs = [np.array([int(x) for x in config_str]) for config_str in config_strings]

    return configs, energy


def Dwave_sol(ham): # classical tabu solver

    n_qubit_ = 0
    for i_, term_ in enumerate(list(ham.keys())):
        if len(term_) == 1:
            n_qubit_+=1 

    index_couple_list = itertools.combinations(list(range(n_qubit_)), 2)
    h = dict(zip(list(range(n_qubit_)),list(ham.values())[:n_qubit_]))
    J = dict(zip(index_couple_list, list(ham.values())[n_qubit_:]))
    
    response = QBSolv().sample_ising(h, J)

    config = []
    for index, item in list(response.samples())[0].items():
        config.append(item)
        
    return [(np.array(config)*-1 +1)/2], list(response.data_vectors['energy'])[0]



def QAOA_calculate(backend, ham, optimizer_maxiter = 5, shots = 1024, print_eval = True):

    n_qubit_ = 0
    for i_, term_ in enumerate(list(ham.keys())):
        if len(term_) == 1:
            n_qubit_+=1 

    #index_couple_list = []
    index_couple_list = list(itertools.combinations(list(range(n_qubit_)), 2))

    h_reindex = {}

    for index, item in enumerate(ham.items()):
        if len(item[0]) == 1: #bias term
            h_reindex[(index,)] = item[1]
        if len(item[0]) == 2: #coupling term
            h_reindex[index_couple_list[index-n_qubit_]] = item[1]
    
    weights = list(h_reindex.values())

    qp_ = weight_to_qp(np.array(weights), n_qubit=n_qubit_)
    
    shot=shots

    quantum_instance = QuantumInstance(backend, shot) # modify this to do error mitigation
    #optimizer = NELDER_MEAD(maxiter=optimizer_maxiter, maxfev=) # SPSA, L_BFGS_B, QNSPSA, NELDER_MEAD
    optimizer = NELDER_MEAD(maxfev=optimizer_maxiter)

    def store_intermediate_result(eval_count, parameters, mean, std):
        if print_eval == True:
            print("qaoa execution count :", eval_count)

    qaoa_mes = QAOA(optimizer = optimizer, reps = 1, quantum_instance = quantum_instance, initial_point=[0., 1.],
                    callback=store_intermediate_result)

    qaoa = MinimumEigenOptimizer(qaoa_mes)
    result = qaoa.solve(qp_)

    return [np.array([int(i) for i in result.x])], energy_finder(h_reindex, list(result.x))





def weight_to_qp(rand_weight, n_qubit):

    h_coeff      = rand_weight[:n_qubit] #[::-1] #np.full(n_qubit, 1) 
    J_spin_coeff = np.zeros((n_qubit, n_qubit))
    J_spin_coeff_ = rand_weight[n_qubit:] #[::-1] #np.full((len(rand_weight[n_qubit:])), 1)
    if np.sum(J_spin_coeff_) != 0:

        c = 0
        for i in range(n_qubit):
            for j in range(n_qubit):
                if i != j :
                    if J_spin_coeff[i][j] == 0 and J_spin_coeff[j][i] == 0:
                        J_spin_coeff[i][j] += J_spin_coeff_[c]
                        #print("c = ", c)
                        c += 1
        
        J = np.zeros((n_qubit, n_qubit))
        for i in range(n_qubit):
            for j in range(n_qubit):
                J[i][j] += J_spin_coeff[i][j]*2
                J[j][i] += J_spin_coeff[i][j]*2

        J_elim = np.sum(J, axis=0)*-1

    elif np.sum(J_spin_coeff_) == 0:
        J = np.zeros((n_qubit, n_qubit))
        J_elim = np.sum(J, axis=0)*-1


    h = h_coeff*-2
    mdl = Model()
    z = [mdl.binary_var() for i in range(n_qubit)]
    objective = \
        mdl.sum(J[i, j] * z[i] * z[j] for i in range(n_qubit)for j in range(n_qubit)) + \
            mdl.sum(J_elim[k]*z[k] for k in range(n_qubit)) + \
                mdl.sum(h[l]*z[l] for l in range(n_qubit))
    mdl.minimize(objective)
    qp = QuadraticProgram()
    qp.from_docplex(mdl)
    

    return qp


def compute_expectation(counts, ham):
    
    avg = 0
    sum_count = 0

    for bitstring, count in counts.items():
        
        #obj = energy_finder(ham, [int(i) for i in list(bitstring[::-1])])
        obj = energy_finder(ham, [int(i) for i in list(bitstring)])
        avg += obj * count
        sum_count += count
        
    return avg/sum_count

def graph_from_hamiltonian(hamiltonian):

    G = nx.Graph()
    for term,weight in hamiltonian.items():
        if(len(term)==1):
            G.add_node(term[0], weight=weight)
        elif(len(term)==2):
            G.add_edge(term[0], term[1], weight=weight)
    return G

def create_qaoa_circ(ham, theta):
    G = graph_from_hamiltonian(ham)
    nqubits = len(G.nodes())
    p = len(theta)//2  # number of alternating unitaries
    qc = QuantumCircuit(nqubits)
    
    beta = theta[:p]
    gamma = theta[p:]
    
    # initial_state
    for i in range(0, nqubits):
        qc.h(i)
    
    for irep in range(0, p):
        
        # problem unitary
        for pair in list(G.edges()):
            qc.rzz(2 * gamma[irep], pair[0], pair[1])
        for bias in list(G.nodes()): 
            qc.rz(2 * gamma[irep], bias)
        # mixer unitary
        for i in range(0, nqubits):
            qc.rx(2 * beta[irep], i)
            
    qc.measure_all()
        
    return qc

def vqe_ansatz(
    ham,
    parameters, #-> 2N + layers*N
    layers = 2,
    entangler = "full" # "full" or "linear"
    ):
    G = graph_from_hamiltonian(ham)
    qubits = list(range(len(G.nodes())))
    parameters = np.reshape(parameters, (layers+1,len(qubits)*2))
    circ = QuantumCircuit(len(qubits))
    for layer in range(layers):

        for iz in range (0, len(qubits)):
            circ.ry(parameters[layer][iz], qubits[iz])
            circ.rz(parameters[layer][iz+len(qubits)], qubits[iz])

        circ.barrier()
        if entangler == "full":
            for i, j in itertools.combinations(list(range(len(qubits))), 2):
                circ.cz(qubits[i], qubits[j])

        elif entangler == "linear":
            for i in range(len(qubits)-1):
                circ.cz(qubits[i], qubits[i+1])

        circ.barrier()

    for iz in range (0, len(qubits)):
        circ.ry(parameters[layers][iz], qubits[iz])
        circ.rz(parameters[layers][iz+len(qubits)], qubits[iz])

    circ.measure_all()
    return circ 

def VQE_calculate(backend, ham, optimizer_maxiter = 5, shots = 1024, print_eval = True, layers = 2, print_energy = True):

    global eval_num
    eval_num = 0
    
    def get_expectation(ham, backend = backend, shots = shots):
        global eval_num
        
        def execute_circ(theta):
            global eval_num
            qc = vqe_ansatz(ham, theta, layers = layers)
            #qc = create_qaoa_circ(ham, theta)
            counts = backend.run(qc, 
                                nshots=shots).result().get_counts()
            eval_num += 1
            if print_eval == True:
                print("eval", eval_num)
            config = [int(i) for i in list(max(counts, key = counts.get))]
            energy = energy_finder(ham, config)
            if print_energy == True:
                print("Expectation = ", compute_expectation(counts, ham))
            return compute_expectation(counts, ham) #energy #compute_expectation(counts, ham)
        
        return execute_circ
    
    G = graph_from_hamiltonian(ham)
    qubits = len(G.nodes())
    expectation = get_expectation(ham, backend = backend, shots = shots)

    res = minimize(expectation, 
                        #[0 for i in range(qubits*2*(layers + 1))], 
                        np.random.rand((layers+1) * (qubits*2)),
                        method='Nelder-Mead', options={'maxiter':optimizer_maxiter})
                        #method='COBYLA', options={'maxiter':optimizer_maxiter})


    qc_res = create_qaoa_circ(ham, res.x)
    #qc_res = vqe_ansatz(ham, res.x)
    
    counts = backend.run(qc_res, nshots=shots).result().get_counts()
    config = [int(i) for i in list(max(counts, key = counts.get))]
    energy = energy_finder(ham, config)
    return  config, energy

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

