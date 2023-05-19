import numpy as np
import matplotlib.pyplot as plt 
import pydtmc, csv
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Layer, Lambda, InputLayer

class symbol:
    def __init__(self, val) -> None:
        self.val = val
    def __repr__(self):
        return f"{self.val}"
    def __add__(self, other):
        if(self.val == "0"):
            return symbol(other.val)
        elif(other.val == "0"):
            return symbol(self.val)
        else:
            return symbol("c")
def gen_transition_matrix(n, p, q):
    # generate TPM for n states
    tpm = []
    for i in range(n):
        row = [0]*(n)
        if(i==0):
            row[0] = q
            row[1] = 1-q 
        elif(i==n-1):
            row[n-2] = 1-q
            row[n-1] = q 
        else:
            row[i-1] = 1-p-q
            row[i] = q
            row[i+1] = p
        tpm.append(row)
    return tpm
def get_steps(num_iters = 100, l = 10, T=3 ,  n_max = 2**8,n_min = 10,  jumps = 5, q=0.2, seeds=[0,1,2]):
    # simulates T types of nodes with 
    states = [str(i) for i in range(n_min, n_max)]
    p=(1-q)/2
    print(f"Length of trial is l={l:.2f}")
    # print(f"Epsilon for trial = {np.log(1 - ((65/l)**0.5))/np.log(0.04):.3f}")
    tpm = gen_transition_matrix(n_max - n_min, p, q)
    tpm = np.linalg.matrix_power(tpm, jumps)
    mc = pydtmc.MarkovChain(tpm, states)
    curr_state = str(int(n_max/2))
    steps=[]
    for b in range(T):
        steps_b = mc.simulate(num_iters, curr_state, seed=seeds[b])
        steps_b = [int(elem) for elem in steps_b]
        steps.append(steps_b)
    return np.array(steps)
def sim_3_ss_bb(l=10, nodes=[6, 6, 6], estimates = [4, 2, 3]):
    # simulate 3-SS-BB with trial of length l blocks, taking (T-1)l slots to execute
    T = len(nodes)
    # There are (T-1) slots in every block 
    # for each node, a block is chosen uniformly at random 
    num_nodes_in_blocks = np.zeros((l, T))
    for b in range(T):
        # Type b nodes
        num_nodes_b = nodes[b]
        blocks_b = np.random.randint(0, l, num_nodes_b)
        # print(blocks_b)
        # num_nodes_in_blocks[blocks_b, b]+=1
        for block in blocks_b:
            num_nodes_in_blocks[block, b]+=1
    bit_pattern = gen_pattern_3_ss_bb(num_nodes_in_blocks, estimates)
    return num_nodes_in_blocks, bit_pattern

def gen_pattern_3_ss_bb(num_nodes_in_blocks, estimates):
    # given number of nodes in blocks and previous estimates, generate a tx pattern according to 3-SS-BB
    l, T = num_nodes_in_blocks.shape
    p_participate = [min(1, 1.6*l/estimate) for estimate in estimates]
    # print(p_participate)
    # bit_pattern = np.zeros((l, T-1))
    bit_pattern = np.array([[symbol('0') for b in range(T-1)] for block in range(l)])
    # generate bit pattern according to the pattern for different types of nodes 
    for block in range(l):
        for b in range(T):
            num_nodes_b_block = int(num_nodes_in_blocks[block, b])
            for i in range(num_nodes_b_block):
                if(b):
                    # type 2, .. , T
                    bit_p = np.array([symbol('0') for slot in range(T-1)])
                    bit_p[b-1] = symbol('b')
                    bit_pattern[block]+=bit_p
                else:
                    # type 1
                    bit_p = np.array([symbol('a') for slot in range(T-1)])
                    bit_pattern[block]+=bit_p
    return bit_pattern_to_string(bit_pattern)

def bit_pattern_to_string(bit_p):
    # returns array of strings instead of objects
    x, y = bit_p.shape
    return np.array([[bit_p[i,j].val for j in range(y)] for i in range(x)])

def bit_pattern_to_onehot(bit_p: np.ndarray)->np.ndarray: 
    # turns the bit pattern to a vector with onehot encoding for 0,a,b,c
    # print(f"Shape of bit pattern = {bit_p.shape}")
    l = bit_p.shape[0]
    T = bit_p.shape[1]+1
    #l, T obtained
    bit_p_flattened = bit_p.flatten()
    # print(f"Flattened vector = {bit_p_flattened}")
    symbol_to_int = {'0':0, 'a':1, 'b':2, 'c':3}
    integer_encoded = [symbol_to_int[sym] for sym in bit_p_flattened]
    # print(f"Integer encoded = {integer_encoded}")
    onehot_encoded = []
    for val in integer_encoded:
        letter = [0]*4
        letter[val] = 1
        onehot_encoded+=letter
    onehot_encoded=np.array(onehot_encoded)
    # print(f"Shape of onehot encoded = {onehot_encoded.shape}")
    return onehot_encoded


def gen_feature_vectors_for_slot(n_max, l=10, nodes=[6,6,6], estimates=[4,2,3], prev_truths=[5, 4, 2]):
    # given length of trial, number of nodes, previous estimates, generate a feature vector for student and teacher
    # last T elements are targets
    num_nodes_in_blocks, bit_pattern = sim_3_ss_bb(l, nodes, estimates)
    bit_p_onehot = bit_pattern_to_onehot(bit_pattern)
    feature_vector_student = np.append(bit_p_onehot, np.array(estimates))
    feature_vector_student = np.append(feature_vector_student, np.array(nodes).flatten())
    # print(f"Feature vector for student = {feature_vector_student.shape}")
    
    feature_vector_teacher = num_nodes_in_blocks.flatten()
    feature_vector_teacher = np.append(feature_vector_teacher, prev_truths)
    feature_vector_teacher = np.append(feature_vector_teacher, np.array(nodes).flatten())
    # print(f"Feature vector for teacher = {feature_vector_teacher.shape}") 
    return feature_vector_student, feature_vector_teacher/n_max

def gen_training_data_teacher_run_sim(tag = "het_test", num_iters = 100, l = 10, T=3 ,  n_max = 2**8,n_min = 10,  jumps = 5, q=0.2, seeds=[0,1,2]):
    steps = get_steps(num_iters, l, T, n_max, n_min, jumps, q, seeds)
    estimates = steps[:, 0]
    print(f"estimates {estimates}")
    prev_truths = steps[:,0]
    
    feature_vec_length = l*T+T
    teacher = Sequential()
    teacher.add(Dense(feature_vec_length, input_shape=(feature_vec_length, ), activation='relu'))
    teacher.add(Dense(int(feature_vec_length*(0.5)), activation='sigmoid'))
    teacher.add(Dense(int(feature_vec_length*(0.5)), activation='sigmoid'))
    teacher.add(Dense(T, activation='linear'))
    teacher.compile(loss='mean_squared_error', optimizer='adam')
    fname = f"./data/{tag}.csv"
    for i in range(num_iters):
        nodes = steps[:,i]
        fv_student, fv_teacher = gen_feature_vectors_for_slot(n_max, l, nodes, estimates, prev_truths)
        target = fv_teacher[-T:]
        X_teacher = fv_teacher[:-T]
        X_teacher = X_teacher.reshape(1, X_teacher.shape[0])
        # print(f"X_teacher shape = {X_teacher.shape}")
        prediction = teacher.predict(X_teacher, verbose = -1)
        print(f"i={i}, pred={prediction}, target={target}")
        estimates = prediction[0]
        prev_truths = nodes
        y_teacher = target.reshape((1, T))
        teacher.fit(X_teacher, y_teacher, verbose=-1)
        with open(fname, 'a') as f:
            writer=csv.writer(f)
            writer.writerow(fv_teacher)
    return
def train_teacher_offline(tag='het_test', epochs = 500, T=3):
    fname = f"./data/{tag}.csv"
    data = np.genfromtxt(fname, delimiter=',')
    np.random.shuffle(data)
    split = 0.9
    num_samples = data.shape[0]
    split_sample = int(split*num_samples)
    print(data.shape)
    X = data[:split_sample, :-T]
    y = data[:split_sample, -T:]
    X_test = data[split_sample:, :-T]
    y_test = data[split_sample:, -T:]
    y_test = np.reshape(y_test, (num_samples-split_sample, T))
    
    feature_vec_length = X.shape[1]
    teacher = Sequential()
    teacher.add(Dense(feature_vec_length, input_shape=(feature_vec_length, ), activation='relu'))
    teacher.add(Dense(int(feature_vec_length*(0.5)), activation='sigmoid'))
    teacher.add(Dense(int(feature_vec_length*(0.5)), activation='sigmoid'))
    teacher.add(Dense(T, activation='linear'))
    teacher.compile(loss='mean_squared_error', optimizer='adam')
    
    history = teacher.fit(X,y, validation_data=(X_test, y_test), epochs = epochs, batch_size = 16, shuffle=True)
    return history

if __name__=='__main__':
    # gen_training_data_teacher_run_sim(num_iters=5000)
    history = train_teacher_offline(epochs = 1000)
    print(history.history.keys())
    plt.plot(history.history['loss'], alpha=0.6)
    plt.plot(history.history['val_loss'], alpha=0.8)
    plt.grid()
    # plt.ylim(0,1e-4)
    plt.yscale('log')
    plt.title('Heterogenous nodes : Teacher loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("./plots/final_loss_log_5000.png")
    plt.show()

    