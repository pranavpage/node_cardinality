import numpy as np
import matplotlib.pyplot as plt 
import pydtmc
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
        print(blocks_b)
        # num_nodes_in_blocks[blocks_b, b]+=1
        for block in blocks_b:
            num_nodes_in_blocks[block, b]+=1
    bit_pattern = gen_pattern_3_ss_bb(num_nodes_in_blocks, estimates)
    return num_nodes_in_blocks, bit_pattern

def gen_pattern_3_ss_bb(num_nodes_in_blocks, estimates):
    # given number of nodes in blocks and previous estimates, generate a tx pattern according to 3-SS-BB
    l, T = num_nodes_in_blocks.shape
    p_participate = [min(1, 1.6*l/estimate) for estimate in estimates]
    print(p_participate)
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

if __name__=='__main__':
    l = 10
    T = 3
    num_nodes_in_blocks, bit_pattern = sim_3_ss_bb()
    print(num_nodes_in_blocks)
    print(bit_pattern)
    print(bit_pattern.flatten())
    alpha = symbol('a')
    arr = np.array([alpha]*4)
    steps = get_steps(l=l, T=T, num_iters=1000)
    for b in range(T):
        plt.plot(steps[b], label = f"Type {b}")
    plt.xlabel("Number of iterations")
    plt.ylabel("Number of nodes")
    plt.grid(True)
    plt.legend()
    plt.show()